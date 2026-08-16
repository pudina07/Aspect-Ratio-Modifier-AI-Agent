"""
Converts and activates all 258+ Agency Agents as native Google Antigravity skills and rules.
Installs to workspace (.agents/skills/), global Antigravity config (~/.gemini/config/skills/),
and creates an AGENTS.md index for automatic discovery.
"""
import os
import re
import sys
import shutil
from pathlib import Path

# Force UTF-8 stdout/stderr for Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR / "agency-agents"
USER_HOME = Path.home()

# Target installation directories for Antigravity
WORKSPACE_SKILLS = BASE_DIR / ".agents" / "skills"
WORKSPACE_RULES = BASE_DIR / ".agents" / "rules"
GLOBAL_GEMINI_SKILLS = USER_HOME / ".gemini" / "config" / "skills"
GLOBAL_CLI_SKILLS = USER_HOME / ".gemini" / "antigravity-cli" / "skills"
INTEGRATION_DIR = REPO_DIR / "integrations" / "antigravity"

DIVISIONS = [
    "academic", "design", "engineering", "finance", "game-development",
    "gis", "healthcare", "marketing", "paid-media", "product",
    "project-management", "sales", "security", "spatial-computing",
    "specialized", "strategy", "support", "testing"
]

def slugify(text: str) -> str:
    """Convert agent name or filename to a clean kebab-case slug."""
    text = text.lower().strip()
    text = re.sub(r'[\s_]+', '-', text)
    text = re.sub(r'[^a-z0-9\-]', '', text)
    text = re.sub(r'-+', '-', text)
    return text.strip('-')

def parse_agent_md(file_path: Path):
    """Parses frontmatter and body from an agent markdown file."""
    content = file_path.read_text(encoding="utf-8", errors="replace")
    
    name = ""
    description = ""
    color = ""
    emoji = "🤖"
    vibe = ""
    body = content

    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1]
            body = parts[2].strip()

            for line in fm_text.splitlines():
                line_clean = line.strip()
                if line_clean.startswith("name:"):
                    name = line_clean[5:].strip().strip('"\'')
                elif line_clean.startswith("description:"):
                    description = line_clean[12:].strip().strip('"\'')
                elif line_clean.startswith("color:"):
                    color = line_clean[6:].strip().strip('"\'')
                elif line_clean.startswith("emoji:"):
                    emoji = line_clean[6:].strip().strip('"\'')
                elif line_clean.startswith("vibe:"):
                    vibe = line_clean[5:].strip().strip('"\'')

    # Fallbacks if name or description missing in frontmatter
    if not name:
        stem = file_path.stem
        # e.g. engineering-frontend-developer -> Frontend Developer
        parts = stem.split('-')
        if len(parts) > 1:
            name = ' '.join(parts[1:]).title()
        else:
            name = stem.replace('-', ' ').title()

    if not description:
        # Grab first heading or first sentence from body
        first_line = body.splitlines()[0] if body.splitlines() else name
        description = f"Specialized {name} agent for professional workflows."

    # Build unique slug
    file_stem = file_path.stem
    if file_stem.startswith("agency-"):
        slug = file_stem
    else:
        # Prefix with agency-
        slug = f"agency-{slugify(name)}"

    return {
        "name": name,
        "slug": slug,
        "description": description,
        "emoji": emoji,
        "color": color,
        "vibe": vibe,
        "body": body,
        "division": file_path.parent.name,
        "source_file": file_path
    }

def format_skill_md(agent: dict) -> str:
    """Formats an agent into a standard Antigravity SKILL.md file."""
    # Ensure description is clean and single-line / properly folded
    desc = agent['description'].replace('\n', ' ').strip()
    
    skill_content = f"""---
name: {agent['slug']}
description: {desc}
---

# {agent['emoji']} {agent['name']} ({agent['division'].title()} Division)

> **Vibe:** {agent['vibe'] if agent['vibe'] else 'Specialized domain expert'}

{agent['body']}
"""
    return skill_content

def install_agents():
    print("=" * 60)
    print("   INSTALLING AGENCY AGENTS AS ANTIGRAVITY SKILLS & RULES   ")
    print("=" * 60)

    # Ensure directories exist
    for target in [WORKSPACE_SKILLS, WORKSPACE_RULES, GLOBAL_GEMINI_SKILLS, GLOBAL_CLI_SKILLS, INTEGRATION_DIR]:
        target.mkdir(parents=True, exist_ok=True)

    installed_agents = []
    by_division = {}

    for div in DIVISIONS:
        div_dir = REPO_DIR / div
        if not div_dir.exists():
            continue
        
        agent_files = sorted(list(div_dir.glob("*.md")))
        by_division[div] = []

        print(f"\nProcessing division '{div}' ({len(agent_files)} agents)...")
        for f in agent_files:
            try:
                agent = parse_agent_md(f)
                skill_text = format_skill_md(agent)

                # Write to all Antigravity skill locations
                for target_base in [WORKSPACE_SKILLS, GLOBAL_GEMINI_SKILLS, GLOBAL_CLI_SKILLS, INTEGRATION_DIR]:
                    agent_skill_dir = target_base / agent['slug']
                    agent_skill_dir.mkdir(parents=True, exist_ok=True)
                    skill_file = agent_skill_dir / "SKILL.md"
                    skill_file.write_text(skill_text, encoding="utf-8")

                installed_agents.append(agent)
                by_division[div].append(agent)
                print(f"  ✓ {agent['emoji']} {agent['slug']} ({agent['name']})")
            except Exception as e:
                print(f"  ✗ Error processing {f.name}: {e}")

    # Generate workspace AGENTS.md and .agents/rules/agency-roster.md
    print("\n--- Generating Master AGENTS.md and Roster Rules ---")
    roster_lines = [
        "# 🏢 The Agency — Agent Roster & Specialized Skills Index",
        "",
        "This project has **The Agency's full roster of 258+ specialized AI agents** activated as native Antigravity skills and rules.",
        "",
        "## How to Activate an Agent",
        "To invoke any agent, ask for it by its slug or role, for example:",
        "- `\"Use the agency-software-architect skill to design our database schema\"`",
        "- `\"Consult agency-security-auditor to review this authentication flow\"`",
        "- `\"Activate agency-frontend-developer to build our responsive UI\"`",
        "",
        "---",
        ""
    ]

    for div, agents in by_division.items():
        roster_lines.append(f"### 📂 {div.upper()} DIVISION ({len(agents)} agents)")
        roster_lines.append("| Agent Slug | Name | Description |")
        roster_lines.append("|---|---|---|")
        for a in agents:
            clean_desc = a['description'].replace('|', '-').replace('\n', ' ')
            roster_lines.append(f"| `{a['slug']}` | {a['emoji']} {a['name']} | {clean_desc} |")
        roster_lines.append("")

    roster_text = "\n".join(roster_lines)
    (BASE_DIR / "AGENTS.md").write_text(roster_text, encoding="utf-8")
    (WORKSPACE_RULES / "agency-roster.md").write_text(roster_text, encoding="utf-8")
    print(f"✓ Saved master roster to {BASE_DIR / 'AGENTS.md'}")
    print(f"✓ Saved rule to {WORKSPACE_RULES / 'agency-roster.md'}")

    print("\n" + "=" * 60)
    print(f"🎉 SUCCESS! {len(installed_agents)} AGENCY AGENTS ACTIVATED ACROSS ALL LOCATIONS:")
    print(f"  1. Workspace Skills: {WORKSPACE_SKILLS}")
    print(f"  2. Global Gemini Config: {GLOBAL_GEMINI_SKILLS}")
    print(f"  3. Antigravity CLI Skills: {GLOBAL_CLI_SKILLS}")
    print(f"  4. Workspace Rulebook: {BASE_DIR / 'AGENTS.md'}")
    print("=" * 60)

if __name__ == "__main__":
    install_agents()
