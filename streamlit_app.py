from __future__ import annotations

import csv
import html as html_module
import io
import re
import urllib.error
import urllib.request
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from notionhelper import NotionHelper
from rapidfuzz import fuzz, process


IMMUNISATION_FIELDS = [
    "DTP",
    "Hep B",
    "Varicella",
    "Measles IgG",
    "Mumps IgG",
    "Rubella IgG",
    "Imms Free Text",
]
MASTER_COURSES_PATH = Path("master_training_courses.csv")


st.set_page_config(
    page_title="Staff training reminders",
    page_icon=":material/school:",
    layout="wide",
)


def require_password() -> None:
    """Gate the app with the password stored in Streamlit secrets."""
    app_secrets = st.secrets.get("app", {})
    expected = app_secrets.get("PASSWORD", "")
    if not expected:
        st.error("Set PASSWORD in .streamlit/secrets.toml before using the app.")
        st.stop()

    if st.session_state.get("authenticated"):
        return

    st.title("Staff training reminders")
    _, login_column, _ = st.columns([1, 1.2, 1])
    with login_column:
        st.caption("Administrator sign-in")
        with st.form("login", border=False):
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", type="primary")
    if submitted:
        if password == expected:
            st.session_state.authenticated = True
            st.rerun()
        st.error("Incorrect password.")
    st.stop()


def read_upload(uploaded_file) -> pd.DataFrame:
    """Read CSV exports including TeamNet's UTF-16/null-byte format."""
    raw = uploaded_file.getvalue()
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "latin-1"):
        try:
            text = raw.decode(encoding)
            if "\x00" not in text:
                break
        except UnicodeDecodeError:
            continue
    text = text.replace("\x00", "")
    return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)


def clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().strip('"')


def find_column(columns: list[str], terms: tuple[str, ...]) -> str | None:
    for column in columns:
        normalized = re.sub(r"[^a-z0-9]", "", column.lower())
        if any(term in normalized for term in terms):
            return column
    return None


def parse_training_export(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Convert the wide TeamNet matrix into one row per staff/course item."""
    df = df.copy()
    df.columns = [clean_text(column) for column in df.columns]
    name_column = df.columns[0]
    df[name_column] = df[name_column].map(clean_text)
    df = df[df[name_column].ne("")]

    courses = [column for column in df.columns[1:] if clean_text(column)]
    records: list[dict[str, str | date | None]] = []
    today = date.today()
    for _, row in df.iterrows():
        staff_name = clean_text(row[name_column])
        for course in courses:
            raw_value = clean_text(row[course])
            if not raw_value:
                continue
            lower = raw_value.lower()
            if "overdue" in lower:
                status = "Overdue"
            elif "not due" in lower:
                status = "Not due"
            else:
                status = "Due"
            parsed_date = pd.to_datetime(
                re.sub(r"\s*\([^)]*\)", "", raw_value),
                dayfirst=True,
                errors="coerce",
            )
            due_date = parsed_date.date() if not pd.isna(parsed_date) else None
            records.append(
                {
                    "Staff member": staff_name,
                    "Training course": clean_text(course),
                    "Raw status": raw_value,
                    "Status": status,
                    "Due date": due_date,
                    "Days until due": (due_date - today).days if due_date else None,
                }
            )
    return pd.DataFrame(records), courses


def load_master_courses(course_names: list[str]) -> pd.DataFrame:
    """Load the editable course/URL list and add newly seen TeamNet courses."""
    columns = ["Course name", "Course URL", "Due every (years)"]
    if MASTER_COURSES_PATH.exists():
        saved = pd.read_csv(MASTER_COURSES_PATH, dtype=str, keep_default_na=False)
        saved.columns = [clean_text(column) for column in saved.columns]
        name_column = next((column for column in saved.columns if column.casefold() in {"course name", "course", "training course"}), None)
        url_column = next((column for column in saved.columns if "url" in column.casefold() or "link" in column.casefold()), None)
        if name_column:
            interval_column = next((column for column in saved.columns if "due every" in column.casefold() or "frequency" in column.casefold() or "interval" in column.casefold()), None)
            master = pd.DataFrame(
                {
                    "Course name": saved[name_column].map(clean_text),
                    "Course URL": saved[url_column].map(clean_text) if url_column else "",
                    "Due every (years)": saved[interval_column].map(clean_text) if interval_column else "",
                }
            )
        else:
            master = pd.DataFrame(columns=columns)
    else:
        master = pd.DataFrame(columns=columns)

    existing = set(master["Course name"].str.casefold())
    additions = [name for name in course_names if name.casefold() not in existing]
    if additions:
        master = pd.concat(
            [master, pd.DataFrame({"Course name": additions, "Course URL": [""] * len(additions), "Due every (years)": [""] * len(additions)})],
            ignore_index=True,
        )
    return master[columns].drop_duplicates(subset=["Course name"], keep="first")


def prepare_emails(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_text(column) for column in df.columns]
    email_column = find_column(list(df.columns), ("email", "mail"))
    name_column = find_column(list(df.columns), ("name", "staff", "employee", "person"))
    # Notion may also expose a relation property called "Staff member". The
    # title property "Name" is the human-readable value we need for matching.
    exact_name_column = next(
        (column for column in df.columns if clean_text(column).casefold() == "name"),
        None,
    )
    name_column = exact_name_column or find_column(list(df.columns), ("name", "staff", "employee", "person"))
    if email_column is None or name_column is None:
        raise ValueError("The Notion directory must contain Name and Email properties.")
    result = df[[name_column, email_column]].rename(
        columns={name_column: "Staff member", email_column: "Email"}
    )
    result["Staff member"] = result["Staff member"].map(clean_text)
    result["Email"] = result["Email"].map(clean_text)
    result = result[(result["Staff member"] != "") & (result["Email"] != "")]
    team_column = find_column(list(df.columns), ("department", "team", "service", "division"))
    if team_column:
        result["Team"] = df.loc[result.index, team_column].map(clean_text)
    else:
        result["Team"] = ""
    for field in IMMUNISATION_FIELDS:
        source_column = next(
            (column for column in df.columns if clean_text(column).casefold() == field.casefold()),
            None,
        )
        if source_column:
            result[field] = df.loc[result.index, source_column].map(clean_text)
        else:
            result[field] = ""
    return result.drop_duplicates(subset=["Staff member"])


def normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9 ]", "", clean_text(value).lower())


def match_staff_to_directory(training_names: pd.Series, directory: pd.DataFrame) -> pd.DataFrame:
    """Fuzzy-match TeamNet names to Notion names and retain confidence details."""
    choices = directory["Staff member"].tolist()
    normalized_choices = {name: normalize_name(name) for name in choices}
    matched_rows = []
    for training_name in sorted(training_names.dropna().unique()):
        normalized_training = normalize_name(training_name)
        exact = next((name for name, normalized in normalized_choices.items() if normalized == normalized_training), None)
        candidates = process.extract(
            training_name,
            choices,
            scorer=fuzz.token_sort_ratio,
            limit=2,
        )
        best_name, best_score = (exact, 100.0) if exact else (candidates[0][0], candidates[0][1])
        second_score = candidates[1][1] if candidates and candidates[0][0] == best_name and len(candidates) > 1 else 0
        confident = bool(best_name) and best_score >= 78 and (best_score - second_score >= 3 or best_score == 100)
        if confident:
            matched = directory[directory["Staff member"].eq(best_name)].iloc[0]
            matched_rows.append(
                {
                    "Staff member": training_name,
                    "Email": matched["Email"],
                    "Team": matched["Team"],
                    **{field: matched[field] for field in IMMUNISATION_FIELDS},
                    "Notion name": best_name,
                    "Match score": round(best_score),
                    "Match status": "Matched",
                }
            )
        else:
            matched_rows.append(
                {
                    "Staff member": training_name,
                    "Email": "",
                    "Team": "",
                    **{field: "" for field in IMMUNISATION_FIELDS},
                    "Notion name": best_name or "",
                    "Match score": round(best_score) if best_name else 0,
                    "Match status": "Needs review",
                }
            )
    return pd.DataFrame(matched_rows)


@st.cache_data(ttl=300, max_entries=4, show_spinner="Loading staff directory from Notion...")
def load_staff_directory(notion_token: str, data_source_id: str) -> pd.DataFrame:
    """Read the staff directory from Notion for the current app session."""
    helper = NotionHelper(notion_token=notion_token, max_retries=3, request_timeout=30.0)
    return helper.get_data_source_pages_as_dataframe(
        data_source_id,
        include_page_ids=False,
        utc=True,
    )


def build_reminders(training: pd.DataFrame, emails: pd.DataFrame, master_courses: pd.DataFrame, days_ahead: int) -> pd.DataFrame:
    today = date.today()
    due = training[
        (training["Status"].eq("Overdue"))
        | (training["Due date"].notna() & training["Due date"].le(today + timedelta(days=days_ahead)))
    ].copy()
    reminders = due.merge(emails, on="Staff member", how="left")
    reminders = reminders.merge(
        master_courses[["Course name", "Course URL"]],
        left_on="Training course",
        right_on="Course name",
        how="left",
    ).drop(columns=["Course name"])
    return reminders[reminders["Match status"].eq("Matched") & reminders["Email"].ne("")]


def send_email(to: str, subject: str, html_body: str, text_body: str) -> tuple[bool, str]:
    app_secrets = st.secrets.get("app", {})
    api_key = app_secrets.get("RESEND_API_KEY", "")
    if not api_key:
        return False, "RESEND_API_KEY is not set in secrets."
    payload = json.dumps(
        {
            "from": "Stanhope Staff Training <hello@attribut.me>",
            "to": [to],
            "reply_to": ["sally.james@nhs.net"],
            "subject": subject,
            "html": html_body,
            "text": text_body,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://api.resend.com/emails",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "StanhopeStaffTraining/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return True, response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return False, error.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as error:
        return False, str(error.reason)


def immunisation_copy(row: pd.Series) -> tuple[str, str]:
    """Return an HTML block and text fallback for one staff member's records."""
    team = clean_text(row.get("Team", "")).casefold()
    hep_b_required = not any(role in team for role in ("admin", "receptionist", "manager"))
    required_fields = IMMUNISATION_FIELDS[:-1] if hep_b_required else [field for field in IMMUNISATION_FIELDS[:-1] if field != "Hep B"]
    complete = all(normalize_bool(row.get(field, "")) for field in required_fields)
    free_text = clean_text(row.get("Imms Free Text", ""))
    if complete:
        notes_html = (
            f'<p style="margin:10px 0 0;color:#475569;"><strong>Additional notes:</strong> {html_module.escape(free_text)}</p>'
            if free_text
            else ""
        )
        return (
            '<div style="margin:24px 0;padding:16px 18px;border:1px solid #b3cbd1;border-radius:12px;background:#b6cfd5;color:#0f172a;font-family:Arial,sans-serif;">'
            '<strong>Immunisation Record Complete</strong>' + notes_html + '</div>',
            "Immunisation Record Complete" + (f"\nAdditional notes: {free_text}" if free_text else ""),
        )

    labels = {"DTP": "DTP", "Hep B": "Hep B", "Varicella": "Varicella", "Measles IgG": "Measles IgG", "Mumps IgG": "Mumps IgG", "Rubella IgG": "Rubella IgG"}
    cells = []
    text_lines = []
    for field in IMMUNISATION_FIELDS[:-1]:
        if field == "Hep B" and not hep_b_required:
            label, value, colour = "Hep B", "Not required for this role", "#64748b"
        else:
            present = normalize_bool(row.get(field, ""))
            label, value, colour = labels[field], ("Record on file" if present else "Record Missing"), ("#166534" if present else "#b84d55")
        cells.append(f'<tr><td style="padding:8px 0;color:#475569;">{label}</td><td style="padding:8px 0;text-align:right;color:{colour};font-weight:600;">{value}</td></tr>')
        text_lines.append(f"{label}: {value}")
    if free_text:
        escaped = html_module.escape(free_text)
        cells.append(f'<tr><td style="padding:8px 0;color:#475569;vertical-align:top;">Additional notes</td><td style="padding:8px 0;text-align:right;color:#334155;">{escaped}</td></tr>')
        text_lines.append(f"Additional notes: {free_text}")
    html_block = '<div style="margin:24px 0;padding:18px 20px;border:1px solid #e2e8f0;border-radius:12px;background:#f8fafc;font-family:Arial,sans-serif;"><h2 style="margin:0 0 10px;color:#0f172a;font-size:17px;">Immunisation records</h2><table style="width:100%;border-collapse:collapse;font-size:14px;">' + "".join(cells) + "</table></div>"
    return html_block, "\n".join(text_lines)


def normalize_bool(value: object) -> bool:
    return clean_text(value).casefold() in {"true", "yes", "y", "1", "checked", "complete", "on"}


def reminder_copy(staff_name: str, rows: pd.DataFrame) -> tuple[str, str, str]:
    subject = "Action required: staff training reminder"
    safe_name = html_module.escape(staff_name)
    training_rows = []
    text_lines = [f"Hello {staff_name}", "", "The following mandatory training is overdue or due soon:", ""]
    for _, row in rows.iterrows():
        due_label = row["Raw status"]
        if row["Due date"]:
            due_label = f"{row['Due date'].strftime('%d/%m/%Y')} ({row['Status'].lower()})"
        course_name = html_module.escape(row["Training course"])
        course_url = clean_text(row.get("Course URL", ""))
        if course_url.startswith(("https://", "http://")):
            course_html = f'<a href="{html_module.escape(course_url, quote=True)}" style="color:#cc808e !important;text-decoration:underline;"><strong>{course_name}</strong></a>'
            text_course = f"{row['Training course']} ({course_url})"
        else:
            course_html = f"<strong>{course_name}</strong>"
            text_course = row["Training course"]
        training_rows.append(f'<li style="margin:8px 0;color:#334155;">{course_html} — {html_module.escape(due_label)}</li>')
        text_lines.append(f"- {text_course}: {due_label}")
    immunisation_html, immunisation_text = immunisation_copy(rows.iloc[0])
    text_lines.extend(["", "Immunisation records:", immunisation_text, "", "Please complete the training as soon as possible.", "", "Kind regards,", "Stanhope Staff Training"])
    html_body = f'''<style>a, a:visited {{ color:#cc808e !important; }}</style><div style="background:#f1f5f9;padding:32px 16px;font-family:Arial,sans-serif;color:#0f172a;">
<div style="max-width:620px;margin:0 auto;background:#ffffff;border-radius:16px;overflow:hidden;box-shadow:0 4px 18px rgba(15,23,42,.08);">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"><tr><td bgcolor="#b84d55" style="background-color:#b84d55;padding:28px 32px;color:#ffffff;"><div style="font-size:12px;letter-spacing:1.5px;text-transform:uppercase;opacity:.85;">Stanhope Staff Training</div><h1 style="margin:10px 0 0;font-size:25px;line-height:1.2;color:#ffffff;">Training reminder</h1></td></tr></table>
<div style="padding:28px 32px;"><p style="font-size:16px;">Hello {safe_name},</p><p style="color:#475569;line-height:1.6;">The following mandatory training is overdue or due soon:</p>
<ul style="padding-left:20px;line-height:1.5;">{"".join(training_rows)}</ul>{immunisation_html}
<p style="color:#475569;line-height:1.6;">Please complete the training as soon as possible.</p><p style="margin-top:28px;">Kind regards,<br><strong>Stanhope Staff Training</strong></p></div></div></div>'''
    return subject, html_body, "\n".join(text_lines)


require_password()

st.title("Staff training reminders")
st.caption("Upload the latest TeamNet training matrix. Staff email addresses and departments are read from Notion for this session only.")

with st.sidebar:
    st.header("Reminder settings")
    days_ahead = st.number_input("Include training due within (days)", min_value=0, max_value=365, value=30)
    test_mode = st.toggle(
        "Testing mode",
        value=False,
        help="Send only the first five staff reminders, all addressed to jan.duplessis@nhs.net.",
    )
    st.divider()
    st.caption(f"Today: {date.today().strftime('%d/%m/%Y')}")
    if st.button("Sign out", icon=":material/logout:"):
        st.session_state.authenticated = False
        st.rerun()

matrix_file = st.file_uploader("TeamNet training matrix", type=["csv"], help="The wide export with one staff member per row and one course per column.")

if not matrix_file:
    st.info("Upload the TeamNet training matrix to generate the reminder list.")
    st.stop()

try:
    training_matrix = read_upload(matrix_file)
    training, courses = parse_training_export(training_matrix)
    app_secrets = st.secrets.get("app", {})
    notion_token = app_secrets.get("NOTION_TOKEN", "")
    data_source_id = app_secrets.get("NOTION_DATA_SOURCE_ID", "")
    if not notion_token or not data_source_id:
        raise ValueError("Set NOTION_TOKEN and NOTION_DATA_SOURCE_ID under [app] in .streamlit/secrets.toml.")
    directory = load_staff_directory(notion_token, data_source_id)
except (ValueError, pd.errors.ParserError) as error:
    st.error(f"Could not read the uploads: {error}")
    st.stop()
except Exception as error:  # noqa: BLE001
    st.error(f"Could not load the staff directory from Notion: {error}")
    st.stop()

with st.expander("Training course master list", icon=":material/menu_book:", expanded=False):
    st.caption("Maintain one course URL per row. New courses from the uploaded TeamNet matrix are added automatically.")
    master_courses = load_master_courses(courses)
    edited_master_courses = st.data_editor(
        master_courses,
        num_rows="dynamic",
        hide_index=True,
        key="master_courses_editor",
        column_config={
            "Course name": st.column_config.TextColumn("Course name", required=True),
            "Course URL": st.column_config.LinkColumn("Course URL", help="Optional link staff can use to complete the course."),
            "Due every (years)": st.column_config.NumberColumn("Due every (years)", min_value=0, step=1, help="How often this course must be completed, in years."),
        },
    )
    if st.button("Save course master list", icon=":material/save:"):
        to_save = edited_master_courses.copy()
        to_save["Course name"] = to_save["Course name"].map(clean_text)
        to_save["Course URL"] = to_save["Course URL"].map(clean_text)
        to_save["Due every (years)"] = pd.to_numeric(to_save["Due every (years)"], errors="coerce")
        to_save = to_save[to_save["Course name"].ne("")].drop_duplicates(subset=["Course name"], keep="first")
        to_save.to_csv(MASTER_COURSES_PATH, index=False)
        st.success(f"Saved {len(to_save)} training courses to {MASTER_COURSES_PATH}.")

with st.expander("Inspect full Notion staff directory", expanded=False):
    st.caption(f"{len(directory)} rows × {len(directory.columns)} columns returned from the configured Notion data source.")
    st.dataframe(directory, hide_index=True, height=500)

email_directory = prepare_emails(directory)
email_list = match_staff_to_directory(training["Staff member"], email_directory)

reminders = build_reminders(training, email_list, master_courses, int(days_ahead))
needs_review = email_list[email_list["Match status"].ne("Matched")]

overdue_count = int(reminders["Status"].eq("Overdue").sum())
staff_count = int(reminders["Staff member"].nunique())
course_count = int(reminders["Training course"].nunique())

metric_columns = st.columns(4)
metric_columns[0].metric("Staff needing reminders", staff_count)
metric_columns[1].metric("Training items", len(reminders))
metric_columns[2].metric("Overdue items", overdue_count)
metric_columns[3].metric("Courses represented", course_count)

st.subheader("Reminder preview")
filter_col, search_col = st.columns([1, 2])
with filter_col:
    selected_status = st.multiselect("Status", ["Overdue", "Due"], default=["Overdue", "Due"])
with search_col:
    search = st.text_input("Search staff or course", placeholder="Start typing to filter", label_visibility="visible")

filtered = reminders[reminders["Status"].isin(selected_status)].copy()
if "Team" in reminders and reminders["Team"].replace("", pd.NA).notna().any():
    selected_teams = st.multiselect("Team or department", sorted(reminders.loc[reminders["Team"].ne(""), "Team"].unique()), default=[])
    if selected_teams:
        filtered = filtered[filtered["Team"].isin(selected_teams)]
if search:
    mask = filtered["Staff member"].str.contains(search, case=False, na=False) | filtered["Training course"].str.contains(search, case=False, na=False)
    filtered = filtered[mask]

display_columns = ["Staff member", "Notion name", "Match score", "Email", "Training course", "Status", "Due date", "Raw status"]
if "Team" in filtered and filtered["Team"].replace("", pd.NA).notna().any():
    display_columns.insert(1, "Team")
st.dataframe(filtered[display_columns], hide_index=True, column_config={"Due date": st.column_config.DateColumn("Due date", format="DD/MM/YYYY")})

if not needs_review.empty:
    st.warning(f"{len(needs_review)} staff member(s) could not be confidently matched to Notion and will be skipped.")
    st.dataframe(needs_review[["Staff member", "Notion name", "Match score", "Match status"]], hide_index=True)

st.subheader("Send reminders")
st.caption("Each staff member receives one email containing all of their overdue or upcoming training.")
send_scope = st.segmented_control("Send to", ["All staff", "Selected staff"], default="All staff")
selected_staff: list[str] = []
if send_scope == "Selected staff":
    selected_staff = st.multiselect(
        "Select staff members",
        sorted(reminders["Staff member"].unique()),
        placeholder="Choose one or more staff members",
    )
    send_reminders = reminders[reminders["Staff member"].isin(selected_staff)]
else:
    send_reminders = reminders
if test_mode:
    st.warning("Testing mode is on: only the first five staff reminders will be sent to jan.duplessis@nhs.net.")
send_disabled = send_reminders.empty
if st.button("Send reminders", type="primary", icon=":material/send:", disabled=send_disabled):
    results = []
    staff_groups = list(send_reminders.groupby("Staff member", sort=True))
    if test_mode:
        staff_groups = staff_groups[:5]
    for staff_name, rows in staff_groups:
        subject, html_body, text_body = reminder_copy(staff_name, rows)
        email = "jan.duplessis@nhs.net" if test_mode else rows["Email"].iloc[0]
        success, detail = send_email(email, subject, html_body, text_body)
        results.append({"Staff member": staff_name, "Email": email, "Result": "Sent" if success else "Failed", "Details": detail})
    st.session_state.send_results = pd.DataFrame(results)

if "send_results" in st.session_state:
    results = st.session_state.send_results
    sent = int(results["Result"].eq("Sent").sum())
    failed = len(results) - sent
    if failed:
        st.error(f"{sent} reminder(s) sent, {failed} failed.")
    else:
        st.success(f"{sent} reminder(s) sent successfully.")
    st.dataframe(results, hide_index=True)
