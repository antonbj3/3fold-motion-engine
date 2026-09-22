# Section 4.4 refresh-schedule source

`refresh_contrast.json` is the measured 63-row same-law refresh comparison on five scene operators. It includes all units and arms; it is an exported measurement, not a new run. `regenerate.py` extracts the 15 SI cells into `refresh_schedule_si.csv` and prints them. `content_sha256` hashes the public measurement after scene aliasing with the timing fields listed in `sha256_note` excluded. The archived source identity is recorded outside the public package. Capped counts remain censored.
