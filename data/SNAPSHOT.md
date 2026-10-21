# Data snapshot

The committed file data/adult.csv.gz is the dataset every step uses, so a clean clone runs without network access. Rebuild it with `python -m src.download_data`; `src.data` refuses to load it if its SHA256 does not match the value below.

## Source
- Dataset: Adult, UCI Machine Learning Repository (dataset id 2), https://archive.ics.uci.edu/dataset/2/adult
- Download URL: https://archive.ics.uci.edu/static/public/2/adult.zip
- Downloaded: 2026-10-21T04:30:01Z
- Archive SHA256: 7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb
- Citation: Becker, B., & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20

## Licence
- Creative Commons Attribution 4.0 International (CC BY 4.0), as stated on the dataset page. It permits sharing and adaptation for any purpose with attribution.
- Attribution for data/adult.csv.gz: Becker, B., & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20. Licensed under CC BY 4.0. Changes: adult.data and adult.test combined, whitespace stripped, trailing "." removed from test labels, source_file column added.

## Snapshot file
- data/adult.csv.gz: adult.data (32,561 rows) and adult.test (16,281 rows, after skipping its "|1x3 Cross validator" header line) combined, 48,842 rows.
- Columns: the 15 UCI columns (age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country, income) plus source_file.
- Written deterministically (gzip with no file name and mtime 0, LF line endings); two independent downloads produced the same hash.
- Snapshot SHA256: `a55c768bf78ab8dd3e22ebd9554de8eebfb92f1713e16500a18c435a3b495ae0`

## Facts that affect modelling
- Label: income above 50K in 11,687 rows (23.9%), at or below 50K in 37,155 rows.
- Missing values are coded "?": workclass 2,799 rows, occupation 2,809 rows, native_country 857 rows; 3,620 rows have at least one. They are kept as a separate category, "Unknown", in each column (see PROTOCOL.md).
- Sex: Male 32,650, Female 16,192.
- Race: White 41,762, Black 4,685, Asian-Pac-Islander 1,519, Amer-Indian-Eskimo 470, Other 406.
- 52 rows are exact duplicates of an earlier row (all 15 UCI columns). They are kept, as in the published data.
- native_country has 42 values (including "?"); occupation has 15.
