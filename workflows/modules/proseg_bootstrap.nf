process ENSURE_PROSEG {
    tag "proseg"

    input:
    val trigger

    output:
    path("proseg_path.txt")

    script:
    def rawSearchPaths = params.proseg_search_paths instanceof List
        ? params.proseg_search_paths
        : params.proseg_search_paths.toString().split(",").collect { searchPath -> searchPath.trim() }
    def searchPathValues = []
    if (params.proseg_binary != null && !(params.proseg_binary instanceof Boolean)) {
        def legacyPath = params.proseg_binary.toString().trim()
        if (legacyPath) {
            searchPathValues << legacyPath
        }
    }
    searchPathValues.addAll(rawSearchPaths.collect { searchPath -> searchPath.toString() })
    def searchPaths = searchPathValues.join("\n")
    """
    set -euo pipefail

    expand_path() {
        local raw="\$1"
        raw="\${raw/#\\~\\//\$HOME/}"
        raw="\${raw//\\\$HOME/\$HOME}"
        printf '%s\\n' "\$raw"
    }

    check_proseg() {
        local candidate="\$1"
        if [ -d "\$candidate" ]; then
            candidate="\$candidate/proseg"
        fi
        if [ -x "\$candidate" ]; then
            "\$candidate" --version >/dev/null
            printf '%s\\n' "\$candidate" > proseg_path.txt
            echo "Using ProSeg: \$candidate" >&2
            "\$candidate" --version >&2
            exit 0
        fi
    }

    cat > proseg_search_paths.txt <<'PATHS'
${searchPaths}
PATHS

    while IFS= read -r configured_path; do
        [ -n "\$configured_path" ] || continue
        check_proseg "\$(expand_path "\$configured_path")"
    done < proseg_search_paths.txt

    if command -v proseg >/dev/null 2>&1; then
        check_proseg "\$(command -v proseg)"
    fi

    if [ "${params.proseg_auto_install}" != "true" ]; then
        echo "ProSeg was not found in configured proseg_search_paths and proseg_auto_install=false." >&2
        exit 1
    fi

    if ! command -v cargo >/dev/null 2>&1; then
        echo "ProSeg was not found and cargo is unavailable; install Rust/cargo or set proseg_install_path to an existing ProSeg binary." >&2
        exit 1
    fi

    install_path="\$(expand_path "${params.proseg_install_path}")"
    install_dir="\$(dirname "\$install_path")"
    mkdir -p "\$install_dir" 2>/dev/null || true

    tmp_root="\$(mktemp -d)"
    trap 'rm -rf "\$tmp_root"' EXIT

    echo "Installing ProSeg with cargo package '${params.proseg_cargo_package}'..." >&2
    cargo install "${params.proseg_cargo_package}" --root "\$tmp_root"

    built_binary="\$tmp_root/bin/proseg"
    if [ ! -x "\$built_binary" ]; then
        echo "cargo install completed but did not produce \$built_binary" >&2
        exit 1
    fi

    if [ -w "\$install_dir" ]; then
        install -m 755 "\$built_binary" "\$install_path"
    else
        echo "Installing ProSeg to \$install_path requires sudo permission." >&2
        sudo -v
        sudo install -m 755 "\$built_binary" "\$install_path"
    fi

    "\$install_path" --version >&2
    printf '%s\\n' "\$install_path" > proseg_path.txt
    """
}
