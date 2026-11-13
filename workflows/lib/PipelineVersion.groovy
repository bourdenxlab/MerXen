import groovy.json.JsonOutput

import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardOpenOption
import java.util.concurrent.TimeUnit

/**
 * Which MerXen release a run used (docs/versioning.md).
 *
 * The release is manifest.version of workflows/nextflow.config. A run from a
 * git checkout also records `git describe` of that checkout, so a run from an
 * untagged or edited checkout is told apart from the release it started from:
 * `v1.2.0` is the release itself, `v1.2.0-3-gabc1234` three commits after it and
 * a `-dirty` suffix uncommitted edits. A run that Nextflow pulled itself
 * (`nextflow run bourdenxlab/MerXen -r v1.2.0`) records the revision and commit.
 */
class PipelineVersion {

    static final String RECORD_FILE = "merxen_versions.jsonl"

    /** `git describe` of the checkout holding the workflow, or null. */
    static String describeCheckout(Path projectDir) {
        if (projectDir == null) {
            return null
        }
        try {
            def command = [
                "git", "-C", projectDir.toString(), "describe",
                "--tags", "--match", "v[0-9]*", "--always", "--dirty",
            ]
            def process = new ProcessBuilder(command).redirectErrorStream(false).start()
            def stdout = new StringBuilder()
            def stderr = new StringBuilder()
            process.consumeProcessOutput(stdout, stderr)
            if (!process.waitFor(10, TimeUnit.SECONDS)) {
                process.destroyForcibly()
                return null
            }
            def text = stdout.toString().trim()
            return process.exitValue() == 0 && text ? text : null
        } catch (Exception ignored) {
            return null
        }
    }

    /** The run's version facts as a flat map (the JSONL record). */
    static Map info(def workflow, String nextflowVersion) {
        def manifest = workflow.manifest
        return [
            merxen_version: manifest?.version?.toString(),
            git_describe: describeCheckout(workflow.projectDir as Path),
            revision: workflow.revision,
            commit_id: workflow.commitId,
            nextflow_version: nextflowVersion,
            run_name: workflow.runName,
            session_id: workflow.sessionId?.toString(),
            resume: workflow.resume,
            stub_run: workflow.stubRun,
            profile: workflow.profile,
            started: workflow.start?.toString(),
        ]
    }

    /** One launch-log line: release plus where the code came from. */
    static String banner(Map info) {
        def source = []
        if (info.revision) {
            source << "revision ${info.revision}"
        }
        if (info.commit_id) {
            source << "commit ${info.commit_id.toString().take(12)}"
        }
        if (!info.revision && !info.commit_id && info.git_describe) {
            source << "checkout ${info.git_describe}"
        }
        def suffix = source ? " (${source.join(', ')})" : ""
        return "MerXen ${info.merxen_version ?: 'unknown version'}${suffix}"
    }

    /** True for `-preview`, which builds the DAG but runs nothing. */
    static boolean isPreview(def workflow) {
        try {
            return workflow.preview as boolean
        } catch (MissingPropertyException ignored) {
            return false
        }
    }

    /**
     * Append the record to <outdir>/nextflow/merxen_versions.jsonl, one line per
     * launch, so a results folder resumed across releases lists each of them.
     * Returns null on success, or the reason the record could not be written.
     */
    static String record(Map info, Path outdir) {
        try {
            def folder = outdir.resolve("nextflow")
            Files.createDirectories(folder)
            Files.writeString(
                folder.resolve(RECORD_FILE),
                JsonOutput.toJson(info) + "\n",
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.APPEND,
            )
            return null
        } catch (Exception error) {
            return error.message ?: error.class.simpleName
        }
    }
}
