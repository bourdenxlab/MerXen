# Cellpose annotation pilot

The versioned [pilot-v1 annotation guide](../../src/merxen/assets/hitl/annotation-guide.md)
is bundled with the installed package and copied into every annotation export.
Use that copy while correcting crops, and retain it with the returned edits.

This initial exporter produces an explicitly unstratified, tissue-spanning
training pilot. It does not generate a held-out test set or establish an
anatomically balanced sampling design. Import and training are later milestones.
