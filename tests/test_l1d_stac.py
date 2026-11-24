"""The L1D identity of the STAC item, the quicklook / thumbnail swap, the transparent PNG quicklook and the KML
that places it on the map (marketplace feedback of 12 and 13 Oct 2026)."""

import copy

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from iride_eaglet_processor.metadata import point_quicklook_at_png, set_l1d_identity
from iride_eaglet_processor.previews import make_l1d_previews, point_kml_at_png

L1C_ID = "IME07_1CSP__RGB3_20261011T094937_20261011T094939_20261011T131246_04354_R022_O_A00"


class Log:
    def __init__(self):
        self.lines = []

    def info(self, message):
        self.lines.append(str(message))


def l1c_stac():
    """the fields of a real L1C STAC item that the processor changes (as in the 10 Sep and 25 Sep products)"""
    preview = f"./PREVIEW/{L1C_ID}.QLK"
    return {
        "id": L1C_ID,
        "collection": "IME-1CSP-RGB3",
        "description": "EAGLET-2 Level 1C 3 bands product",
        "links": [{"rel": "collection", "title": "EAGLET-2 L1C 3 bands product information",
                   "href": "https://{host}/{collections}/{collection}"}],
        "properties": {"processing:level": "L1C",
                       "processing:lineage": L1C_ID.replace("_1CSP_", "_1BSP_"),
                       "processing:software": {"Eaglet-2 Level 1C processor": "02.01"},
                       "processing:version": "Processing baseline: A00"},
        "assets": {
            "thumbnail": {"href": preview + ".png", "type": "image/png", "title": "Thumbnail image",
                          "file:checksum": "1220aa", "file:size": 8, "file:local_path": f"{L1C_ID}/PREVIEW/x.png",
                          "roles": ["thumbnail"]},
            "quicklook": {"href": preview + ".jpg", "type": "image/jpg", "title": "Quicklook image",
                          "file:checksum": "1220bb", "file:size": 2, "file:local_path": f"{L1C_ID}/PREVIEW/x.jpg",
                          "roles": ["overview", "visual"]},
        },
    }


def test_l1d_identity_as_in_the_marketplace_template():
    stac = l1c_stac()
    set_l1d_identity(stac, L1C_ID, "01.00", Log())
    p = stac["properties"]
    assert stac["collection"] == "IME-1DSP-RGB3"
    assert stac["description"] == "EAGLET-2 Level 1D 3 bands product"
    assert stac["links"][0]["title"] == "EAGLET-2 L1D 3 bands product information"
    assert p["processing:level"] == "L1D"
    assert p["processing:lineage"] == L1C_ID                       # the input L1C, with its own dates
    assert p["processing:software"] == {"Eaglet-2 Level 1D processor": "01.00"}
    assert p["processing:version"] == "Processing baseline: A00"   # unchanged


@pytest.mark.parametrize("change", [
    lambda s: s["properties"].update({"processing:level": "L1D"}),      # already an L1D: not run twice
    lambda s: s.update({"collection": "IME-1BSP-RGB3"}),
    lambda s: s.update({"description": "EAGLET-2 3 bands product"}),
    lambda s: s.update({"links": []}),
])
def test_l1d_identity_stops_on_an_unexpected_stac(change):
    stac = l1c_stac()
    change(stac)
    with pytest.raises(ValueError):
        set_l1d_identity(stac, L1C_ID, "01.00", Log())


def test_quicklook_points_to_the_png_and_thumbnail_to_the_jpeg():
    stac = l1c_stac()
    before = copy.deepcopy(stac["assets"])
    point_quicklook_at_png(stac, Log())
    quick, thumb = stac["assets"]["quicklook"], stac["assets"]["thumbnail"]
    assert quick["href"].endswith(".png") and quick["type"] == "image/png"
    assert thumb["href"].endswith(".jpg") and thumb["type"] == "image/jpg"
    for key in ("file:checksum", "file:size", "file:local_path"):          # the file facts move with the file
        assert quick[key] == before["thumbnail"][key] and thumb[key] == before["quicklook"][key]
    assert quick["title"] == "Quicklook image" and quick["roles"] == ["overview", "visual"]   # stay with the asset
    assert thumb["title"] == "Thumbnail image" and thumb["roles"] == ["thumbnail"]
    with pytest.raises(ValueError):                                         # the L1D layout is not swapped back
        point_quicklook_at_png(stac, Log())


def test_png_quicklook_is_transparent_where_there_is_no_data(tmp_path):
    h, w = 120, 160
    yy, xx = np.mgrid[0:h, 0:w]
    inside = (xx > 20 + 0.3 * yy) & (xx < 130 + 0.2 * yy)
    l1d = np.stack([1000 + 50 * k + (xx * 7 + yy * 3) % 900 for k in range(3)]).astype("uint16")
    l1d[:, ~inside] = 0
    image = tmp_path / "l1d.tif"
    with rasterio.open(image, "w", driver="GTiff", height=h, width=w, count=3, dtype="uint16", nodata=0,
                       crs="EPSG:32633", transform=from_origin(500000, 4500000, 2, 2)) as dst:
        dst.write(l1d)
    preview = tmp_path / "PREVIEW"
    preview.mkdir()
    Image.new("RGB", (80, 60)).save(preview / "x.QLK.png")              # as in the L1C: RGB, black no data
    Image.new("RGB", (40, 30)).save(preview / "x.QLK.jpg", quality=90)

    make_l1d_previews(tmp_path, image, Log())

    with rasterio.open(image) as src:
        valid = (src.read(out_shape=(3, 60, 80), resampling=rasterio.enums.Resampling.average) > 0).all(0)
    with Image.open(preview / "x.QLK.png") as png:
        assert png.mode == "RGBA" and png.size == (80, 60)
        rgba = np.array(png)
    assert set(np.unique(rgba[..., 3])) == {0, 255}
    assert np.array_equal(rgba[..., 3] == 255, valid)                   # alpha 0 exactly where there is no data
    with Image.open(preview / "x.QLK.jpg") as jpg:
        assert jpg.format == "JPEG" and jpg.mode == "RGB" and jpg.size == (40, 30)


KML = ('<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2"><GroundOverlay>'
       "<Icon><href>{}</href></Icon><LatLonBox><north>40.55</north><south>40.40</south></LatLonBox>"
       "</GroundOverlay></kml>\n")


def test_kml_places_the_transparent_png(tmp_path):
    preview = tmp_path / "PREVIEW"
    preview.mkdir()
    (preview / "x.QLK.png").write_bytes(b"png")
    (preview / "x.QLK.kml").write_text(KML.format("x.QLK.jpg"))
    point_kml_at_png(tmp_path, Log())
    assert (preview / "x.QLK.kml").read_text() == KML.format("x.QLK.png")      # nothing else changes


def test_kml_that_already_places_the_png_is_kept(tmp_path):
    preview = tmp_path / "PREVIEW"
    preview.mkdir()
    (preview / "x.QLK.png").write_bytes(b"png")
    (preview / "x.QLK.kml").write_text(KML.format("x.QLK.png"))
    point_kml_at_png(tmp_path, Log())
    assert (preview / "x.QLK.kml").read_text() == KML.format("x.QLK.png")


@pytest.mark.parametrize("href, files", [
    ("x.QLK.jpg", []),                     # no PNG to place
    ("x.QLK.tif", ["x.QLK.png"]),          # neither the JPEG nor the PNG
])
def test_kml_stops_on_an_unexpected_layout(tmp_path, href, files):
    preview = tmp_path / "PREVIEW"
    preview.mkdir()
    for name in files:
        (preview / name).write_bytes(b"png")
    (preview / "x.QLK.kml").write_text(KML.format(href))
    with pytest.raises(ValueError):
        point_kml_at_png(tmp_path, Log())


def test_kml_with_two_images_stops(tmp_path):
    preview = tmp_path / "PREVIEW"
    preview.mkdir()
    (preview / "x.QLK.png").write_bytes(b"png")
    (preview / "x.QLK.kml").write_text(KML.format("x.QLK.jpg").replace("</Icon>", "</Icon><Icon><href>y.jpg</href></Icon>"))
    with pytest.raises(ValueError):
        point_kml_at_png(tmp_path, Log())
