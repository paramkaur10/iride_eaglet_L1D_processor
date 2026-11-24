# test_l1d_product.py: the L1D product's metadata: the XML and the STAC describe the L1D image.
import hashlib
import json

import pytest

from iride_eaglet_processor.calibration import CalibrationError, read_xml_calibration, set_l1d_image_in_xml
from iride_eaglet_processor.colour_correction import L1D_GAIN, L1D_OFFSET
from iride_eaglet_processor.metadata import COG_MEDIA_TYPE, compute_sha256, describe_l1d_image, refresh_file_info

L1C_XML = """<?xml version="1.0" encoding="UTF-8"?>\r
<n1:Level-1C_User_Product xmlns:n1="urn:test">\r
  <Checksum algorithm="MD5"><Image>0123abcd</Image></Checksum>\r
  <Checksum algorithm="SHA256"><Image>{old_sha}</Image></Checksum>\r
  <SOLAR_IRRADIANCE Band_ID="1" Unit="W/m²/µm">1288.345</SOLAR_IRRADIANCE>\r
  <L1C_Product_Image_Characteristics>\r
    <Rescaling_Factors Band_ID="1"><GAIN unit="1/DN">1.62019221805984E-05</GAIN><OFFSET>-6.557707036607915E-04</OFFSET></Rescaling_Factors>\r
  </L1C_Product_Image_Characteristics>\r
  <L1C_Product_Image_Characteristics>\r
    <Rescaling_Factors Band_ID='2'>\r
      <GAIN>1.3992538505613578E-05</GAIN>\r
      <OFFSET>0.001990793157538951</OFFSET>\r
    </Rescaling_Factors>\r
    <Rescaling_Factors Band_ID="3"><GAIN>1.8717640881898817E-05</GAIN><OFFSET>0.0023442647290774555</OFFSET></Rescaling_Factors>\r
  </L1C_Product_Image_Characteristics>\r
</n1:Level-1C_User_Product>\r
"""


def test_xml_describes_the_l1d_image_and_nothing_else_changes(tmp_path, logger):
    old_sha = "06EDF2" + "A" * 58
    xml = tmp_path / "product.xml"
    xml.write_bytes(L1C_XML.format(old_sha=old_sha).encode("utf-8"))
    image = tmp_path / "image.tif"
    image.write_bytes(b"the L1D COG")
    set_l1d_image_in_xml(xml, image, L1D_GAIN, L1D_OFFSET, logger)

    assert read_xml_calibration(xml) == {b: (0.0001, -0.1) for b in (1, 2, 3)}
    new_sha = hashlib.sha256(b"the L1D COG").hexdigest().upper()           # same format as before: upper-case hex
    expected = (L1C_XML.format(old_sha=new_sha)
                .replace("1.62019221805984E-05", "0.0001").replace("-6.557707036607915E-04", "-0.1")
                .replace("1.3992538505613578E-05", "0.0001").replace("0.001990793157538951", "-0.1")
                .replace("1.8717640881898817E-05", "0.0001").replace("0.0023442647290774555", "-0.1"))
    assert xml.read_bytes() == expected.encode("utf-8")                     # CRLF, namespaces, MD5 block kept
    assert not list(tmp_path.glob("*.part"))


def test_xml_without_all_three_bands_is_refused_and_left_alone(tmp_path, logger):
    xml = tmp_path / "product.xml"
    text = L1C_XML.format(old_sha="ab" * 32).replace('Band_ID="3"', 'Band_ID="4"')
    xml.write_bytes(text.encode("utf-8"))
    (tmp_path / "image.tif").write_bytes(b"x")
    with pytest.raises(CalibrationError, match="Band_ID"):
        set_l1d_image_in_xml(xml, tmp_path / "image.tif", L1D_GAIN, L1D_OFFSET, logger)
    assert xml.read_bytes() == text.encode("utf-8")


def make_stac(root, image):
    meta = root / "product.xml"
    meta.write_text("<xml/>")
    stac = {
        "id": "P", "properties": {"renders": {"rgb": {"assets": ["rgb"], "rescale": [[1557.0, 9000.0]] * 3,
                                                      "nodata": 9}}},
        "assets": {
            "rgb": {"href": "./IMG_DATA/image.tif", "type": "image/tiff; application=geotiff",
                    "file:checksum": "1220" + "0" * 64, "file:size": 1, "roles": ["data"],
                    "bands": [{"name": n, "eo:common_name": n, "raster:scale": 1.6e-05, "raster:offset": 0.002,
                               "statistics": {"minimum": 1}} for n in ("red", "green", "blue")]},
            "metadata": {"href": "./product.xml", "file:checksum": "0" * 64, "file:size": 1},
            "remote": {"href": "https://example.org/x.tif", "file:checksum": "keep"},
        }}
    path = root / "product.json"
    path.write_text(json.dumps(stac))
    return path, stac


def test_stac_rgb_asset_describes_the_l1d_image(tmp_path, logger):
    (tmp_path / "IMG_DATA").mkdir()
    image = tmp_path / "IMG_DATA" / "image.tif"
    image.write_bytes(b"L1D")
    _, stac = make_stac(tmp_path, image)
    info = dict(method="HYBRID", aux_file="IME___AXE1D_CLUT_x.xml", rescale=[[1100, 3000], [1150, 2900], [1200, 2800]])
    describe_l1d_image(stac, info, logger)
    rgb = stac["assets"]["rgb"]
    assert rgb["type"] == COG_MEDIA_TYPE and "HYBRID" in rgb["description"] and "LZW" in rgb["description"]
    assert all(b["raster:scale"] == 0.0001 and b["raster:offset"] == -0.1 and b["nodata"] == 0
               and b["data_type"] == "uint16" and "statistics" not in b for b in rgb["bands"])
    assert stac["properties"]["renders"]["rgb"]["rescale"] == info["rescale"]
    assert stac["properties"]["renders"]["rgb"]["nodata"] == 0


def test_checksums_and_sizes_follow_the_files(tmp_path, logger):
    (tmp_path / "IMG_DATA").mkdir()
    image = tmp_path / "IMG_DATA" / "image.tif"
    image.write_bytes(b"L1D image bytes")
    path, _ = make_stac(tmp_path, image)
    refresh_file_info(path, logger)
    assets = json.loads(path.read_text())["assets"]
    assert assets["rgb"]["file:checksum"] == compute_sha256(image)                 # multihash kept
    assert assets["rgb"]["file:size"] == len(b"L1D image bytes")
    assert assets["metadata"]["file:checksum"] == compute_sha256(tmp_path / "product.xml")[4:]   # plain hex kept
    assert assets["remote"]["file:checksum"] == "keep"