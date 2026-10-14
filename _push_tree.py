import json
import os
import subprocess
import urllib.request

REPO = "sunzhe12/ua-bench"
BRANCH = "main"
MSG = ("Add the UA-Bench evaluation framework (testsite/)\n\n"
       "Merged from the local release repository, keeping its 9-commit\n"
       "history: loading, answer parsing, gating, scoring, model backends,\n"
       "statistics, three-turn orchestration, reporting and tests.\n"
       "Also brings .gitignore.")

out = subprocess.run(
    ["D:/Git/bin/git.exe", "credential", "fill"],
    input="protocol=https\nhost=github.com\nusername=sunzhe12\n\n",
    capture_output=True, text=True).stdout
TOKEN = [l.split("=", 1)[1] for l in out.splitlines() if l.startswith("password=")][0]


def api(path, method="GET", data=None):
    req = urllib.request.Request("https://api.github.com" + path, method=method)
    req.add_header("Authorization", "Bearer " + TOKEN)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    body = None
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, body, timeout=120) as r:
        raw = r.read()
        return json.loads(raw) if raw else {}


ref = api("/repos/%s/git/ref/heads/%s" % (REPO, BRANCH))
parent_sha = ref["object"]["sha"]
base_tree = api("/repos/%s/git/commits/%s" % (REPO, parent_sha))["tree"]["sha"]

entries = []
skipped = []
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d != ".git"]
    for name in files:
        full = os.path.join(root, name)
        rel = os.path.relpath(full, ".").replace("\\", "/")
        try:
            with open(full, "r", encoding="utf-8") as fh:
                content = fh.read()
        except (UnicodeDecodeError, OSError) as e:
            skipped.append((rel, str(e)))
            continue
        entries.append({
            "path": rel,
            "mode": "100755" if name.endswith(".sh") else "100644",
            "type": "blob",
            "content": content,
        })

tree = api("/repos/%s/git/trees" % REPO, "POST",
           {"base_tree": base_tree, "tree": entries})
commit = api("/repos/%s/git/commits" % REPO, "POST",
             {"message": MSG, "tree": tree["sha"], "parents": [parent_sha]})
api("/repos/%s/git/refs/heads/%s" % (REPO, BRANCH), "PATCH",
    {"sha": commit["sha"]})

print("files in tree : %d" % len(entries))
print("skipped       : %s" % (skipped or "none"))
print("commit        : %s" % commit["sha"][:10])
print("tree          : %s" % tree["sha"][:10])
