from __future__ import annotations

import awkward as ak
import pytest
import uproot

import odapt as od

skhep_testdata = pytest.importorskip("skhep_testdata")


def HZZ_test():
    f = uproot.open(skhep_testdata.data_path("uproot-HZZ.root"))
    original = f["events"].arrays()
    od.root_to_parquet(
        in_file=skhep_testdata.data_path("uproot-HZZ.root"),
        out_file="/Users/zobil/Documents/odapt/test.parquet",
        step_size="100 MB",
        force=True,
    )
    from_parquet = ak.from_parquet("/Users/zobil/Documents/odapt/test.parquet")
    for key in f["events"].keys():
        assert ak.all(from_parquet[key] == original[key])


def specify_tree():
    f = uproot.open(skhep_testdata.data_path("uproot-HZZ.root"))
    original = f["events"].arrays()
    od.root_to_parquet(
        in_file=skhep_testdata.data_path("uproot-HZZ.root"),
        out_file="test.parquet",
        tree="events",
        step_size=200,
        force=True,
    )
    from_parquet = ak.from_parquet("/Users/zobil/Documents/odapt/test.parquet")
    for key in f["events"].keys():
        assert ak.all(from_parquet[key] == original[key])
    # Check row-group size?
    assert (
        ak.metadata_from_parquet("/Users/zobil/Documents/odapt/test.parquet")[
            "num_row_groups"
        ]
        == 13
    )


def Zmumu_test():
    f = uproot.open(skhep_testdata.data_path("uproot-Zmumu.root"))
    original = f["events"].arrays()
    od.root_to_parquet(
        in_file=skhep_testdata.data_path("uproot-Zmumu.root"),
        out_file="test1.parquet",
        step_size="100 MB",
    )
    from_parquet = ak.from_parquet("/Users/zobil/Documents/odapt/test1.parquet")
    for key in f["events"].keys():
        assert ak.all(from_parquet[key] == original[key])


def no_trees():
    with pytest.raises(AttributeError):
        od.root_to_parquet(
            in_file=skhep_testdata.data_path("uproot-hepdata-example.root"),
            out_file="test2.parquet",
            step_size="100 MB",
        )


def two_trees():
    with pytest.raises(AttributeError):
        od.root_to_parquet(
            in_file=skhep_testdata.data_path("uproot-hepdata-example.root"),
            out_file="test2.parquet",
            step_size="100 MB",
        )


specify_tree()
