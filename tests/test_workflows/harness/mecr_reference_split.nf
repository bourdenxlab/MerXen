// Appended by test_mecr_reference_split.py after an include of the real MECR
// processes and the Groovy functions of workflows/main.nf, then the MECR
// wiring of the main.nf workflow block (MECR_WIRING below, cut between text
// anchors). A fake `merxen` in the launch directory's bin/ records each
// task's config JSON instead of computing MECR.
//
// --harness_rows picks the samplesheet: "xenium" (one XENIUM row) or "mixed"
// (the XENIUM row and an ATERA row). Each process output is printed as
//
//     OUT <TAB> <PROCESS> <TAB> <output dir> [<TAB> <pair id> <TAB> <segmentation>] <TAB> END

workflow {
    def rows = [
        [pairId: "PAIR_B", platform: "XENIUM", segmentation: "reseg", zarr: "xenium.zarr"],
    ]
    if (params.harness_rows == "mixed") {
        rows << [pairId: "PAIR_A", platform: "ATERA", segmentation: "original_seg", zarr: "atera.zarr"]
    }
    mecr_samples_ch = Channel.fromList(
        rows.collect { row ->
            tuple(
                row.pairId,
                row.segmentation,
                samplesJsonForSegmentation(
                    row.pairId,
                    [row.platform],
                    [(row.platform): row.zarr],
                    row.segmentation,
                ),
            )
        }
    )
//MECR_WIRING
    // Each line starts on a line of its own and ends with END: task log lines
    // may run into it.
    mecr_reference_results_ch.view { "\nOUT\tMECR_REFERENCE\t${it}\tEND" }
    mecr_atera_reference_results_ch.view { "\nOUT\tMECR_REFERENCE_ATERA\t${it}\tEND" }
    mecr_results_ch.view { pairId, segmentation, out ->
        "\nOUT\tMECR\t${out}\t${pairId}\t${segmentation}\tEND"
    }
}
