import io
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)


# --------------------------------------------------
# CSV EXPORT
# --------------------------------------------------

def create_csv(data):
    """
    Convert prediction results into CSV format.
    Returns CSV data as bytes.
    """

    if data is None or data.empty:
        raise ValueError("No data available for CSV export.")

    return data.to_csv(index=False).encode("utf-8")


# --------------------------------------------------
# PDF REPORT
# --------------------------------------------------

def create_pdf_report(data, title="Customer Churn Prediction Report"):
    """
    Create a PDF report from a pandas DataFrame.
    Returns PDF data as bytes.
    """

    if data is None or data.empty:
        raise ValueError("No data available for PDF export.")

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        spaceAfter=15
    )

    normal_style = ParagraphStyle(
        "ReportText",
        parent=styles["Normal"],
        fontSize=9
    )

    elements = []

    # --------------------------------------------------
    # TITLE
    # --------------------------------------------------

    elements.append(
        Paragraph(title, title_style)
    )

    elements.append(
        Paragraph(
            f"Total Records: {len(data):,}",
            normal_style
        )
    )

    elements.append(Spacer(1, 15))

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    if "Prediction" in data.columns:

        predicted_churn = int(
            (data["Prediction"] == 1).sum()
        )

        predicted_non_churn = int(
            (data["Prediction"] == 0).sum()
        )

        summary_data = [
            ["Prediction Summary", "Count"],
            ["Likely to Churn", predicted_churn],
            ["Not Likely to Churn", predicted_non_churn]
        ]

        summary_table = Table(
            summary_data,
            colWidths=[180, 100]
        )

        summary_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.black
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "ALIGN",
                    (1, 1),
                    (-1, -1),
                    "CENTER"
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    6
                )
            ])
        )

        elements.append(summary_table)

        elements.append(Spacer(1, 20))

    # --------------------------------------------------
    # DATA TABLE
    # --------------------------------------------------

    report_data = data.copy()

    # Make probability easier to read
    if "Churn Probability" in report_data.columns:

        report_data["Churn Probability"] = (
            report_data["Churn Probability"]
            .apply(
                lambda x: f"{x * 100:.2f}%"
                if pd.notna(x)
                else ""
            )
        )

    # Convert values to strings
    report_data = report_data.astype(str)

    # Limit columns to prevent an excessively wide PDF
    max_columns = 10

    if len(report_data.columns) > max_columns:

        report_data = report_data.iloc[:, :max_columns]

    table_data = [
        list(report_data.columns)
    ]

    table_data.extend(
        report_data.values.tolist()
    )

    # Limit rows for PDF readability
    max_rows = 100

    if len(table_data) > max_rows + 1:

        table_data = (
            table_data[:max_rows + 1]
        )

    # Calculate column width
    page_width = landscape(A4)[0] - 60

    column_count = len(table_data[0])

    column_width = page_width / column_count

    report_table = Table(
        table_data,
        repeatRows=1,
        colWidths=[column_width] * column_count
    )

    report_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                6
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.3,
                colors.grey
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "ALIGN",
                (0, 0),
                (-1, -1),
                "CENTER"
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                4
            )
        ])
    )

    elements.append(report_table)

    # --------------------------------------------------
    # FOOTER NOTE
    # --------------------------------------------------

    elements.append(Spacer(1, 15))

    if len(data) > max_rows:

        elements.append(
            Paragraph(
                f"Note: The PDF displays the first "
                f"{max_rows} records. Download the CSV "
                f"for the complete dataset.",
                normal_style
            )
        )

    # --------------------------------------------------
    # BUILD PDF
    # --------------------------------------------------

    document.build(elements)

    buffer.seek(0)

    return buffer.getvalue()