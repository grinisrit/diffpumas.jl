# Independent focused-revision delivery check

`final_bundle_independent_check.json` is the authoritative check of the permanent paper bundle. All 39 archive inputs match paper source commit `1dd8431d74afbad42e62ed005d301881d93f0176`; all 81 pages (33 main, 48 supplement) match the authoritative PDFs in extracted text and rendered pixels. ZIP CRC/hash, both dependency sets, proof render hashes, author contact and build logs pass. `external_bundle_check.json` preserves the identical initial check before temporary delivery output was retired.

`check_focused_delivery.py` reproduces this read-only audit against the permanent bundle; it does not compile, extract source trees or rerun simulations. It is separate from the scientific evidence and the fresh Astra/Consensus exposition review. No journal submission or colored-change baseline verification is implied.
