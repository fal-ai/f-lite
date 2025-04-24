import tomllib

with open("pyproject.toml", "rb") as f:
    data = tomllib.load(f)

deps = data.get("project", {}).get("dependencies", [])
with open("requirements.txt", "w") as out:
    out.write("\n".join(deps) + "\n")