# Windows file-replacement hotfix and checkpoint-safe resume

## Reported failure

The reported final-v4 run reaches CHOA generation 52 / committed FE 1590, then fails
replacing timing.json.tmp with timing.json in last100/validation/CHOA/seed_17/val_1399.
This is a filesystem PermissionError (WinError 5). PowerShell's NativeCommandError is
the presentation of Python's stderr, not a separate CUDA failure. The traceback does
not identify which process or permission caused the denial.

Windows can refuse replacement if another handle does not permit deletion/renaming,
if the destination is read-only, or if directory/file permissions deny the operation.
A scanner, synchronization client or open-file reader is a possible cause, not a
confirmed diagnosis. The previous writer tried os.replace once and aborted. A second
timing write during exception cleanup could then hide the original exception.

## Changes

- Retry only Windows error codes 5, 32 and 33, up to ten attempts with bounded
  exponential backoff (5.55 seconds of sleep in total). Other filesystem errors
  propagate immediately. Permanent access failures still produce an explicit error.
- Use distinct temporary files for JSON and Torch saves in the destination directory,
  close handles before replacement, and clean up the new temporary file on failure
  where possible. The old destination is never explicitly deleted or truncated.
- Apply the same replacement retry to alternating best-vector snapshots.
- Persist running timer snapshots at most once every five seconds; entry and exit
  still persist. Model, Adam epoch and optimizer generation checkpoint cadence is
  unchanged. A hard kill can lose active-time accounting since the last timer save.
- Preserve a training exception if saving its interrupted timer also fails.

No architecture, numerical algorithm, objective, data, threshold, seed, FE budget or
optimizer state version changes. This hotfix does not change Windows permissions,
disable antivirus, remove read-only attributes, reset experiments or move outputs.

## Strict existing-experiment compatibility

The published final profile is based on commit 6e9725f. Its identity normally hashes
all thesis source modules. Simply editing common.py would therefore reject the
existing experiment even though the fix is operational.

Runtime_Source_Compatibility.json contains explicit SHA-256 aliases for only the
reviewed common.py, pipeline.py and optimizers.py revisions. Both Git LF and Windows
CRLF variants map to the corresponding bytes of that published final release. Every
other file remains hashed normally. An additional edit to even an aliased module
stops matching and changes the identity. Data/config/seed/budget changes still change
the identity. This is not a general bypass of fingerprint verification.

No saved key is rewritten. Existing completed runs, frozen choices/protocols, Adam
checkpoints, populations and best vectors remain associated with the original
scientific experiment. check_resume.py performs read-only identity and required-file
checks before training starts; it does not validate every numerical weight value.

## Windows update and resume

Use these steps only after the Python training process has stopped. Keep the existing
outputs/final_v4/main directory, including .lock, state.json, population buffers,
best vectors and shared warm checkpoints. Do not rename the root or manually edit
fingerprints. Reuse the original Windows account, config and output-root override.

Inside the project folder and the existing psmux session:

```powershell
git fetch origin
git switch main
git pull --ff-only origin main
.\.venv-gpu\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-gpu\Scripts\python.exe check_resume.py --config configs/final.json
```

After tests pass and the audit reports identity OK, repeat the original launch:

```powershell
.\.venv-gpu\Scripts\python.exe -u run.py all --config configs/final.json 2>&1 | Tee-Object -FilePath final_v4.log -Append
```

This direct interpreter path does not require Activate.ps1. Completed runs are skipped.
Generation checkpoints resume committed work; only an uncommitted partial generation
may be replayed. The printed generation=52 line follows a successful checkpoint commit,
so the reported run is expected to resume FE 1590 if those files remain intact. The
read-only audit shows the saved generation/FE for actual pending optimizer checkpoints.
Do not start another training copy while a previous one is still active.

If a different output-root override was used, pass that same override to both audit
and training. On an identity error, retain the outputs and investigate the discrepancy;
do not change their saved keys to force acceptance.

## If access denial persists

Retries address transient contention, not permanent ACL or read-only restrictions.
Inspect the specific file while training is stopped:

```powershell
$timingFile = Join-Path (Get-Location) "outputs\final_v4\main\runs\last100\validation\CHOA\seed_17\val_1399\timing.json"
Get-Item -LiteralPath $timingFile | Select-Object FullName, Attributes, IsReadOnly
icacls $timingFile
```

Use the original override path if applicable. Close software holding that file and
check synchronization/security-software logs. A read-only attribute or ACL restriction
needs its specific cause corrected; the patch intentionally does not grant broad
permissions, delete the target or disable protection. Preserve the full traceback and
audit output if the denial continues.

## Verification

Regression checks cover transient denial, bounded permanent denial, retention of the
previous destination, cleanup/serialization errors, Torch checkpoint round trips,
timer throttling, original-exception preservation, exact final-release identities,
LF/CRLF aliases, rejection of unapproved scientific edits, read-only auditing and
exact CHOA replay after a timing/progress failure. The pre-existing suite also checks
all six optimizers' interruption replay and the complete synthetic pipeline.

A native Windows test creates an actual handle denying FILE_SHARE_DELETE and releases
it during retries. It is skipped on Linux; simulated WinError tests run on either
platform. The development container does not establish the actual server's lock owner
or reproduce its antivirus, filesystem synchronization and ACL configuration.

On 2026-10-24, the complete 43-test suite completed on Linux/Python 3.12/PyTorch
2.6.0+cpu: 42 passed, one native-Windows test skipped (13.763 seconds). Source review
confirmed the optimizer edit changes only the replacement helper/import; network,
data, metrics, analysis and final scientific configurations remain byte-identical
to 6e9725f. The explicit LF and CRLF identity aliases and rejection of unapproved
source/data/settings changes passed their regression checks.

Microsoft reference:
https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-deletefilew
