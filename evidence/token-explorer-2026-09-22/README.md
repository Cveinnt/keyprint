# Committed-token explorer validation

Actual MLX/Qwen ordinary and marked generations: 55 and 49 committed tokens,
including end tokens. Traces reconstruct each response exactly. An actual
Transformers/SmolLM2 generation also reconstructs exactly at its 24-token cap.
The first MLX attempt completed inference but its receipt script used the wrong
JSON nesting; that private attempt was retained. The recorded rerun was not
chosen for signal strength.

203 Python checks passed, three skipped, followed by 36 related API checks and
17 focused checks after the final test additions. Nineteen JavaScript checks and
four site packaging checks passed. Wheel contents match all 89 source files.
Browser verification covers token selection, response switching, keyboard Home,
replay, desktop 1280px and mobile 390px without horizontal overflow. No browser
console errors were observed. Local screenshots remain in the receipt directory.

Public data contains text, token IDs and bytes. It excludes keys, random draws
and private journals. This validates trace mechanics, not prose quality,
indistinguishability, detector calibration, or new runtime compatibility. The
website has not been deployed; the SDK remains unpublished and PR remains draft.
