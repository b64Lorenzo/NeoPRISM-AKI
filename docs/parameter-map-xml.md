# XML configuration and XML Notepad

`config/pipeline.xml` contains the complete runtime catalog and settings; no workbook or source sheet is needed. `config/pipeline.template.xml` is the disabled starter, and the single `pipeline.xsd` validates both complete run documents and standalone `<parameterMap>` files.

## Catalog structure

The parameter children are `settings`, `preOperative`, `duringOperative`, `postOperative`, in that order. Shared settings have direct names such as `selectionEnabled`, `exportTimeSeries`, `missingValueTokens`, `derivedFeatures`, `outcomeRules`. Complex rule/parameter values retain typed `entry` objects; provenance fields and unused description tables have been removed.

Each phase owns its end event, optional selection controls, and explicit parameter/class definitions:

```xml
<preOperative selected="true">
  <endEvent>
    <fields><field>event</field></fields>
    <phrases><phrase>start surgery</phrase></phrases>
  </endEvent>
  <includeIds><id>age</id></includeIds>
  <class name="epic.DEMOGRAPHICS">
    <timed>false</timed>
    <parameter id="age">
      <entry name="source_fields"><array><item><string>epic.DEMOGRAPHICS.age</string></item></array></entry>
      <!-- Other required parameter metadata, as in the complete template. -->
    </parameter>
  </class>
</preOperative>
```

This abbreviated excerpt is not a full run document. `settings/selectionEnabled` controls whether phase filters apply. Empty/omitted include/exclude lists mean no restriction; filters affect only their owning phase. `selected` defaults to true. There is no separate selection node/package.

Every parameter is defined once; ID and phase are structural, not duplicated typed entries. Classes group only explicitly defined matching source fields, not all CSV headers. `measurement` and `outcome_criteria` belong to the post-operative parameter. Missing criteria remain unclassified. Measurement fields inherit the parameter's source list only when absent; an explicit empty array means no source. `measurementOrder` keeps export columns stable. Class-wide failure is optional in post-operative classes only and requires review.

Pre-operative end marks operation start, during-operative end marks operation end, and post-operative end marks the configured stay endpoint. Event headers/phrases are editable per phase. Untimed demographics use `<timed>false</timed>` on the pre-operative class; dated extraction still respects the cutoff, and otherwise requires one distinct non-missing value. The during-operative 94-class inventory is visualization-only, not automatic feature selection.

See the [pipeline guide](xml-pipeline-configuration.md) for exact settings, phase-relative series, boundary rows, duration definitions, plotting, models and migration. Runtime semantic checks supplement the XSD; schema validity does not prove data availability or clinical validity.

## Install XML Notepad on Windows

XML Notepad is an optional editor, not a Python or CLI dependency.

1. Ensure .NET Framework 4.8 is installed, as listed in XML Notepad's official setup instructions.
2. Open the [official XML Notepad install page](https://microsoft.github.io/XmlNotepad/install/) and choose the ClickOnce installer for the simplest setup. For offline installation, download the standalone ZIP there, extract it, and run `XmlNotepadSetup.msi`. The [official GitHub releases](https://github.com/microsoft/XmlNotepad/releases/latest) also provide a Windows MSIX installer under **Assets**.
3. Open `config/pipeline.xml` in XML Notepad. Keep `config/pipeline.xsd` beside it; the XML points to the schema with `xsi:noNamespaceSchemaLocation="pipeline.xsd"`. XML Notepad can validate while editing and show errors in its Error List.
4. Save the file and run the project tests:

```powershell
py -3 -m unittest discover -s tests -v
```

To validate both the active configuration and starter template from the project root when `xmllint` is available:

```text
xmllint --noout --schema config/pipeline.xsd config/pipeline.xml config/pipeline.template.xml
```

The project does not install XML Notepad automatically. Use the official project site or its official GitHub releases for downloads. The project XML has no DTD; do not enable DTD processing for untrusted XML files.
