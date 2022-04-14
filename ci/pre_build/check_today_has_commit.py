from datetime import datetime, timezone
import git

repo = git.Repo(".")
currentTime = datetime.now(timezone.utc)
# if currentTime - last commit time more than 26 hours, then return error
if (currentTime - repo.head.commit.committed_datetime).total_seconds() > 26 * 60 * 60:
    print("false")
    exit()
print("true")
