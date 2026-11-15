import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

from src.parameter_map import load_parameter_map, write_parameter_map

ROOT = Path(__file__).resolve().parents[1]
PARAMETER_MAP = ROOT / "config" / "pipeline.xml"


class ParameterMapXmlTests(unittest.TestCase):
    def test_embedded_definitions_are_owned_by_phase_without_duplicate_catalog(self):
        for name in ("pipeline.xml", "pipeline.template.xml"):
            with self.subTest(name=name):
                root = ET.parse(ROOT / "config" / name).getroot()
                relevant = root.find("preprocessing/relevantParameters")
                self.assertEqual([node.tag for node in relevant], ["settings", "pre", "during", "post"])
                settings = relevant.find("settings")
                self.assertIsNone(settings.find("entry[@name='parameters']"))
                self.assertIsNone(settings.find("entry[@name='postoperative_measurements']"))
                definitions = [node for phase in ("pre", "during", "post")
                               for child in relevant.find(phase) if child.tag != "availableClasses"
                               for node in (list(child.findall("parameter")) if child.tag == "class" else [child])]
                self.assertEqual(len(definitions), 21)
                self.assertEqual(len({node.get("id") for node in definitions}), 21)
                self.assertTrue(all(node.tag == "parameter" and node.find("entry[@name='phase']") is None
                                    and node.find("entry[@name='id']") is None for node in definitions))
                death = next(node for node in definitions if node.get("id") == "recorded_death_days")
                self.assertIsNotNone(death.find("entry[@name='measurement']"))
                self.assertIsNotNone(death.find("entry[@name='outcome_criteria']"))

    def test_runtime_map_loads_pre_during_and_postoperative_parameters(self):
        mapping = load_parameter_map(PARAMETER_MAP)
        self.assertEqual(mapping["included_phases"], ["Pre-operative", "During-operative", "Post-operative"])
        self.assertEqual(mapping["catalog_phases"], ["Pre-operative", "During-operative", "Post-operative"])
        self.assertEqual({item["phase"] for item in mapping["parameters"]},
                         {"Pre-operative", "During-operative", "Post-operative"})
        self.assertEqual(sum(item["phase"] == "Pre-operative" for item in mapping["parameters"]), 9)
        self.assertEqual(sum(item["phase"] == "During-operative" for item in mapping["parameters"]), 1)
        self.assertEqual(sum(item["phase"] == "Post-operative" for item in mapping["parameters"]), 11)

    def test_demographics_class_declares_timestamps_unavailable(self):
        for name in ("pipeline.xml", "pipeline.template.xml"):
            with self.subTest(name=name):
                path = ROOT / "config" / name
                demographic_class = ET.parse(path).getroot().find(
                    "preprocessing/relevantParameters/pre/class[@name='epic.DEMOGRAPHICS']"
                )
                self.assertEqual(demographic_class.findtext("timed"), "false")
                mapping = load_parameter_map(path)
                self.assertIn("epic.DEMOGRAPHICS", mapping["timeless_class_names"])
                self.assertFalse(next(p for p in mapping["parameters"] if p["id"] == "age")["timed"])
                self.assertTrue(next(p for p in mapping["parameters"] if p["id"] == "during_creatinine")["timed"])

    def test_during_class_inventory_is_loaded_for_operation_timeline(self):
        inventories = []
        for name in ("pipeline.xml", "pipeline.template.xml"):
            mapping = load_parameter_map(ROOT / "config" / name)
            classes = mapping.get("during_class_names", [])
            inventories.append(classes)
            with self.subTest(name=name):
                self.assertEqual(len(classes), 94)
                self.assertEqual(len({value.casefold() for value in classes}), 94)
                self.assertEqual(classes, sorted(classes, key=str.casefold))
                self.assertIn("epic.EVENTS", classes)
                self.assertIn("epic.LAB", classes)
                self.assertIn("epic.AN-07_intellivu", classes)

        self.assertEqual(inventories[0], inventories[1])

    def test_invalid_during_class_inventory_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid-classes.xml"
            tree = ET.parse(ROOT / "config/pipeline.template.xml")
            inventory = tree.getroot().find("preprocessing/relevantParameters/during/availableClasses")
            ET.SubElement(inventory, "className").text = "epic.LAB"
            tree.write(path, encoding="utf-8", xml_declaration=True)
            with self.assertRaisesRegex(ValueError, "cannot contain duplicates"):
                load_parameter_map(path)

            tree = ET.parse(ROOT / "config/pipeline.template.xml")
            inventory = tree.getroot().find("preprocessing/relevantParameters/during/availableClasses")
            inventory[0].text = "LAB"
            tree.write(path, encoding="utf-8", xml_declaration=True)
            with self.assertRaisesRegex(ValueError, "exact Epic class names"):
                load_parameter_map(path)

    def test_runtime_map_preserves_explicit_class_for_time_series_plots(self):
        mapping = load_parameter_map(PARAMETER_MAP)
        parameters = {item["id"]: item for item in mapping["parameters"]}
        self.assertEqual(parameters["during_creatinine"]["class_name"], "epic.LAB")
        self.assertNotIn("class_name", parameters["operation_diagnosis_proxy"])

    def test_operation_boundaries_are_configured_not_hardcoded(self):
        for name in ("pipeline.xml", "pipeline.template.xml"):
            mapping = load_parameter_map(ROOT / "config" / name)
            boundaries = mapping["phase_boundaries"]
            self.assertEqual(boundaries["operation_start"]["event_fields"], ["epic.EVENTS.EVENT", "event", "events"])
            self.assertIn("start anesthesie", boundaries["operation_start"]["phrases"])
            self.assertEqual(boundaries["operation_end"]["phrases"], [
                "Afronden zorg", "Stop anesthesie", "Vertrek naar IC (of_Afdeling)",
            ])
            self.assertNotIn("preoperative_start_event_patterns", mapping)

    def test_phase_end_events_are_children_of_each_included_phase(self):
        for name in ("pipeline.xml", "pipeline.template.xml"):
            with self.subTest(name=name):
                root = ET.parse(ROOT / "config" / name).getroot()
                settings = root.find("preprocessing/relevantParameters/settings")
                self.assertIsNone(settings.find("entry[@name='phase_boundaries']"))
                self.assertIsNone(settings.find("entry[@name='catalog_phases']"))
                items = settings.find("entry[@name='included_phases']/array").findall("item/object")
                self.assertEqual([item.find("entry[@name='phase']/string").text for item in items],
                                 ["Pre-operative", "During-operative", "Post-operative"])
                self.assertEqual(items[0].find("entry[@name='end_event']/object/entry[@name='phrases']/array/item/string").text,
                                 "start anesthesie")
                self.assertEqual([node.text for node in items[1].findall("entry[@name='end_event']/object/entry[@name='phrases']/array/item/string")],
                                 ["Afronden zorg", "Stop anesthesie", "Vertrek naar IC (of_Afdeling)"])
                self.assertEqual(
                    [node.text for node in items[2].findall("entry[@name='end_event']/object/entry[@name='phrases']/array/item/string")],
                    ["admission end", "Getekend bezoek", "hospital discharge", "discharge from hospital",
                     "discharge home", "ontslag naar huis", "ontslag uit ziekenhuis"],
                )
                mapping = load_parameter_map(ROOT / "config" / name)
                self.assertEqual(mapping["phase_boundaries"]["operation_start"]["event_fields"],
                                 ["epic.EVENTS.EVENT", "event", "events"])
                self.assertEqual(mapping["phase_boundaries"]["postoperative_end"]["phrases"][0],
                                 "admission end")

    def test_postoperative_end_and_stay_duration_rules_are_xml_configured(self):
        for name in ("pipeline.xml", "pipeline.template.xml"):
            with self.subTest(name=name):
                mapping = load_parameter_map(ROOT / "config" / name)
                boundary = mapping["phase_boundaries"]["postoperative_end"]
                self.assertEqual(boundary["event_fields"], ["epic.EVENTS.EVENT", "event", "events"])
                self.assertIn("hospital discharge", boundary["phrases"])
                self.assertIn("admission end", boundary["phrases"])
                self.assertIn("Getekend bezoek", boundary["phrases"])
                measurements = {item["id"]: item for item in mapping["postoperative_measurements"]}
                icu = measurements["icu_los"]["duration"]
                self.assertEqual(icu["method"], "field_interval")
                self.assertEqual(icu["aggregation"], "sum_overlaps")
                self.assertEqual(icu["start_field"], "q_umcg_aki_marco.adm_ic_admdate")
                self.assertEqual(icu["end_field"], "q_umcg_aki_marco.adm_ic_disdate")
                self.assertEqual(icu["match_start_boundary"], "operation_start")
                self.assertEqual(icu["match_end_boundary"], "postoperative_end")
                self.assertEqual(icu["datetime_format"], "iso8601")
                self.assertEqual(icu["timezone"], "Europe/Amsterdam")
                self.assertEqual(icu["unit"], "days")
                hospital = measurements["hospital_los"]["duration"]
                self.assertEqual(hospital["method"], "phase_boundary_interval")
                self.assertEqual(hospital["aggregation"], "difference")
                self.assertEqual(hospital["start_boundary"], "operation_end")
                self.assertEqual(hospital["end_boundary"], "postoperative_end")
                self.assertEqual(hospital["unit"], "days")

    def test_invalid_phase_item_boundary_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            tree = ET.parse(ROOT / "config/pipeline.template.xml")
            items = tree.getroot().findall(
                "preprocessing/relevantParameters/settings/entry[@name='included_phases']/array/item/object"
            )
            phrases = items[1].find("entry[@name='end_event']/object/entry[@name='phrases']/array")
            phrases.clear()
            path = Path(temporary) / "invalid.xml"
            tree.write(path, encoding="utf-8", xml_declaration=True)
            with self.assertRaisesRegex(ValueError, "phase_boundaries.*operation_end.*phrases"):
                load_parameter_map(path)
            ET.SubElement(ET.SubElement(phrases, "item"), "string").text = "Stop anesthesie"
            end_event = items[2].find("entry[@name='end_event']")
            for child in list(end_event):
                end_event.remove(child)
            ET.SubElement(end_event, "string").text = "unexpected"
            tree.write(path, encoding="utf-8", xml_declaration=True)
            with self.assertRaisesRegex(ValueError, "phase_boundaries.postoperative_end needs event_fields and phrases"):
                load_parameter_map(path)

    def test_incomplete_operation_boundary_config_is_rejected(self):
        from src.parameter_map import write_parameter_map
        with tempfile.TemporaryDirectory() as temporary:
            mapping = load_parameter_map(ROOT / "config/pipeline.template.xml")
            mapping["phase_boundaries"]["operation_end"]["phrases"] = []
            path = Path(temporary) / "incomplete.xml"
            write_parameter_map(path, mapping)
            with self.assertRaisesRegex(ValueError, "phase_boundaries.*operation_end.*phrases"):
                load_parameter_map(path)

    def test_measurement_order_and_source_inheritance_remain_stable(self):
        mapping = load_parameter_map(PARAMETER_MAP)
        self.assertEqual([item["id"] for item in mapping["postoperative_measurements"]], [
            "postop_creatinine_72h", "postop_urine_72h", "aki_kdigo_stage",
            "ventilation_duration", "icu_los", "hospital_los", "mortality_30d",
            "recorded_death_days", "postop_ecmo", "prism_ii", "pim",
        ])
        specs = {item["id"]: item for item in mapping["postoperative_measurements"]}
        self.assertEqual(specs["postop_creatinine_72h"]["source_fields"],
                         ["epic.LAB.Kreatinine (umol/L)"])
        self.assertEqual(specs["icu_los"]["source_fields"], [])
        self.assertNotIn("source_fields", specs["aki_kdigo_stage"])

    def test_recorded_death_days_is_user_added_postoperative_metadata(self):
        mapping = load_parameter_map(PARAMETER_MAP)
        parameter = next(item for item in mapping["parameters"] if item["id"] == "recorded_death_days")
        measurement = next(item for item in mapping["postoperative_measurements"] if item["id"] == "recorded_death_days")
        self.assertEqual(parameter["phase"], "Post-operative")
        self.assertEqual(parameter["source_fields"], ["epic.DEMOGRAPHICS.days mortality"])
        self.assertEqual(parameter["workbook_row"], None)
        self.assertIsNone(parameter["outcome_criteria"]["success"])
        self.assertEqual(parameter["outcome_criteria"]["failure"], {"operator": "gte", "value": 0})
        self.assertEqual(measurement["record_scope"], "patient")
        self.assertIsNone(measurement["window_hours"])

    def test_round_trip_preserves_nested_xml_values_and_types(self):
        expected = {
            "text": "A & B < C",
            "integer": 3,
            "decimal": 2.5,
            "enabled": True,
            "unset": None,
            "array": ["x", 4, {"name": "nested"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.xml"
            write_parameter_map(path, expected)
            self.assertEqual(load_parameter_map(path, validate_runtime=False), expected)

    def test_duplicate_xml_entry_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.xml"
            path.write_text(
                '<parameterMap formatVersion="1"><object>'
                '<entry name="x"><string>1</string></entry>'
                '<entry name="x"><string>2</string></entry>'
                '</object></parameterMap>',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate XML entry"):
                load_parameter_map(path, validate_runtime=False)

    def test_single_schema_describes_pipeline_and_parameter_map_elements(self):
        self.assertFalse((ROOT / "data" / "relevant_parameters.xsd").exists())
        schema = ET.parse(ROOT / "config" / "pipeline.xsd").getroot()
        xs = "{http://www.w3.org/2001/XMLSchema}"
        top_level = {
            element.get("name")
            for element in schema.findall(f"{xs}element")
        }
        self.assertEqual(top_level, {"pipelineConfiguration", "parameterMap"})
        elements = list(schema.iter(f"{xs}element"))
        undocumented = [
            element.get("name")
            for element in elements
            if element.find(f"{xs}annotation/{xs}documentation") is None
        ]
        self.assertEqual(undocumented, [])
        for kind in ("complexType", "simpleType"):
            undocumented_types = [
                node.get("name")
                for node in schema.findall(f"{xs}{kind}")
                if node.get("name") and node.find(f"{xs}annotation/{xs}documentation") is None
            ]
            self.assertEqual(undocumented_types, [])

    def test_schema_and_template_files_exist(self):
        self.assertTrue((ROOT / "config" / "pipeline.xsd").is_file())
        self.assertTrue((ROOT / "config" / "pipeline.template.xml").is_file())


if __name__ == "__main__":
    unittest.main()
