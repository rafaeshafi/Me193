# ME 193 – AI & Robotics

Course workspace, one git repo: <https://github.com/rafaeshafi/Me193>
Each project/assignment lives in its own subfolder.

Open this folder (`~/ME193`) in VS Code to see everything at once.

## Projects

| Folder | What it is |
|---|---|
| `lego-single-motor/` | Day 2 – Python script that spins a single LEGO Education motor over BLE (`legoeducation` library) |
| `P1-Pose-Race/` | **P1: Pose Race** – drive a LEGO car by moving your arms. MediaPipe pose tracking over BLE, plus a pose classifier trained on your own recordings |
| `lego-docs/` | Reference clone of <https://github.com/LEGO/LEGOEducation> – API docs + official examples (gitignored) |

## Notes

- Each project keeps its own virtual environment (`my_env/`), gitignored and rebuilt from that project's `requirements.txt`:

  ```sh
  cd <project> && python3 -m venv my_env && my_env/bin/pip install -r requirements.txt
  ```

- `lego-docs/` is a plain clone of LEGO's reference repo, not a dependency. Re-pull it with
  `git clone https://github.com/LEGO/LEGOEducation.git lego-docs` (or `git -C lego-docs pull`).
  The library itself comes from PyPI (`legoeducation`), currently 1.1.1.
- Accounts close **Dec 10** — push work to GitHub and keep an md summary current (this file / per-project READMEs).
