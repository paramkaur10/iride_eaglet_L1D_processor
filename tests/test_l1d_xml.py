"""The L1D header of the product XML, as in Enrico's L1D template of 12 Oct 2026."""

from datetime import datetime, timezone

import pytest

from iride_eaglet_processor.product_xml import set_l1d_header_in_xml

L1C = "IME07_1CSP__RGB3_20261011T094937_20261011T094939_20261011T131246_04354_R022_O_A00"
L1D = "IME07_1DSP__RGB3_20261011T094937_20261011T094939_20261012T151500_04354_R022_O_A00"
GEN = datetime(2026, 10, 12, 15, 15, 0, 123456, tzinfo=timezone.utc)      # the time in the L1D name
AUX = "IME___AXE1D_CLUT_20240101T000000_20990101T000000_20261007T090320____________O_001"
L1A_AUX = "IME07_AXE1A______20240101T000000_20990101T000000_20250722T081128____________O_001"


class Log:
    def __init__(self):
        self.lines = []

    def info(self, message):
        self.lines.append(str(message))


def l1c_xml(product_id=L1C, level="L1C", aux=(L1A_AUX,), inputs=(L1C.replace("_1CSP_", "_1BSP_") + ".zip",)):
    """the parts of a real L1C XML that matter here, laid out as in the products"""
    aux_lines = "".join(f"\n    <AUXILIARY_FILE>{a}</AUXILIARY_FILE>" for a in aux)
    input_lines = "".join(f'\n      <PRODUCT elem="{i + 1}">{p}</PRODUCT>' for i, p in enumerate(inputs))
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n<Level-1C_Main_Product_Header>\n'
            f"  <PRODUCT_ID>{product_id}</PRODUCT_ID>\n"
            f"  <SPACECRAFT_NAME>IRIDE-MS1-EAGLET2-07</SPACECRAFT_NAME>\n"
            f"  <PROCESSING_LEVEL>{level}</PROCESSING_LEVEL>\n"
            f"  <PROCESSING_BASELINE>A00</PROCESSING_BASELINE>\n"
            f"  <PRODUCT_DOI>https://www.aresys.it</PRODUCT_DOI>\n"
            f"  <Processing_Info>\n"
            f"    <GENERATION_TIME>2026-10-11T13:12:46.123456Z</GENERATION_TIME>{aux_lines}\n"
            f"    <DEM>ellipsoid</DEM>\n"
            f"    <INPUT_PRODUCTS>{input_lines}\n    </INPUT_PRODUCTS>\n"
            f"    <Processor_Version>02.01</Processor_Version>\n"
            f"  </Processing_Info>\n"
            f"  <L1C_Product_Image_Characteristics>\n"
            f'    <SOLAR_IRRADIANCE Band_ID="1" Unit="W/m²/µm">1288.345</SOLAR_IRRADIANCE>\n'
            f'    <Rescaling_Factors Band_ID="1">\n      <GAIN>0.0001</GAIN>\n      <OFFSET>-0.1</OFFSET>\n'
            f"    </Rescaling_Factors>\n"
            f"  </L1C_Product_Image_Characteristics>\n"
            f'  <Checksum algorithm="SHA256">\n    <Image>6785ce3c</Image>\n  </Checksum>\n'
            f"</Level-1C_Main_Product_Header>\n")


def run(tmp_path, text):
    xml = tmp_path / f"{L1C}.xml"
    xml.write_bytes(text.encode("utf-8"))
    set_l1d_header_in_xml(xml, L1C, L1D, GEN, f"{L1C}.zip", tmp_path / "aux" / f"{AUX}.xml", "01.00", Log())
    return xml.read_bytes().decode("utf-8")


def test_header_as_in_the_template(tmp_path):
    before = l1c_xml(aux=(L1A_AUX, L1A_AUX.replace("AXE1A", "AXE1B")))
    after = run(tmp_path, before)
    expected = (before.replace(f"<PRODUCT_ID>{L1C}<", f"<PRODUCT_ID>{L1D}<")
                .replace("<PROCESSING_LEVEL>L1C<", "<PROCESSING_LEVEL>L1D<")
                .replace("<GENERATION_TIME>2026-10-11T13:12:46.123456Z<", "<GENERATION_TIME>2026-10-12T15:15:00.123456Z<")
                .replace(f"<AUXILIARY_FILE>{L1A_AUX}<", f"<AUXILIARY_FILE>{AUX}<")
                .replace(f"\n    <AUXILIARY_FILE>{L1A_AUX.replace('AXE1A', 'AXE1B')}</AUXILIARY_FILE>", "")
                .replace(f'<PRODUCT elem="1">{L1C.replace("_1CSP_", "_1BSP_")}.zip<', f'<PRODUCT elem="1">{L1C}.zip<')
                .replace("<Processor_Version>02.01<", "<Processor_Version>01.00<"))
    assert after == expected          # nothing else changes: element names, units, layout


def test_input_products_become_the_one_l1c(tmp_path):
    after = run(tmp_path, l1c_xml(inputs=("a.zip", "b.zip")))
    assert after.count("<PRODUCT elem=") == 1 and f'<PRODUCT elem="1">{L1C}.zip</PRODUCT>' in after


@pytest.mark.parametrize("text", [
    l1c_xml(product_id=L1C.replace("IME07", "IME03")),                 # the XML of another product
    l1c_xml(level="L1D"),                                              # already an L1D
    l1c_xml(aux=()),                                                   # no AUX file listed
    l1c_xml(inputs=()),                                                # no input product listed
    l1c_xml().replace("<INPUT_PRODUCTS>", '<INPUT_PRODUCTS count="1">'),   # a layout we have not seen
    l1c_xml().replace("  <Processing_Info>\n", "").replace("  </Processing_Info>\n", ""),
    l1c_xml().replace("<GENERATION_TIME>2026-10-11T13:12:46.123456Z</GENERATION_TIME>", ""),   # no generation time
])
def test_unexpected_xml_stops_the_processor(tmp_path, text):
    xml = tmp_path / f"{L1C}.xml"
    xml.write_bytes(text.encode("utf-8"))
    with pytest.raises(ValueError):
        set_l1d_header_in_xml(xml, L1C, L1D, GEN, f"{L1C}.zip", tmp_path / f"{AUX}.xml", "01.00", Log())
    assert xml.read_bytes() == text.encode("utf-8")                   # the file is not touched


def test_generation_time_must_be_utc(tmp_path):
    xml = tmp_path / f"{L1C}.xml"
    xml.write_bytes(l1c_xml().encode("utf-8"))
    with pytest.raises(ValueError):
        set_l1d_header_in_xml(xml, L1C, L1D, GEN.replace(tzinfo=None), f"{L1C}.zip", tmp_path / f"{AUX}.xml",
                              "01.00", Log())
