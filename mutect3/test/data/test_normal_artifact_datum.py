from mutect3.data import normal_artifact_datum


def test_normal_artifact_datum():
    normal_alt_count = 3
    normal_depth = 77
    tumor_alt_count = 8
    tumor_depth = 50
    downsampling = 0.1
    variant_type = "SNV"

    datum = normal_artifact_datum.NormalArtifactDatum(normal_alt_count, normal_depth, tumor_alt_count, tumor_depth,
                 downsampling, variant_type)

    assert datum.normal_depth() == normal_depth
    assert datum.normal_alt_count() == normal_alt_count
    assert datum.tumor_depth() == tumor_depth
    assert datum.tumor_alt_count() == tumor_alt_count
    assert datum.downsampling() == downsampling
    assert datum.variant_type() == variant_type
