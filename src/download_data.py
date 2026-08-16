import GEOparse

accessions = ["GSE59999", "GSE114134", "GSE114135"]

for acc in accessions:
    gse = GEOparse.get_GEO(geo=acc, destdir="../data")
    print(acc, "-> samples:", len(gse.gsms), "| platform:", list(gse.gpls.keys()))
    first_sample = list(gse.gsms.values())[0]
    print(first_sample.metadata.get("characteristics_ch1"))
    print("---")
