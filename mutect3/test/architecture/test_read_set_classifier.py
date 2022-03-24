from mutect3.test.test_utils import artificial_data


def test_separate_gaussian_data():
    data = artificial_data.make_two_gaussian_data(10000)