import os
import sys
import requests


with open(sys.argv[1]) as fp:
    lines = fp.readlines()


flag = False
start = 0
end = 0


for i, line in enumerate(lines):
    if "MODELS_HASH" in line:
        flag = True
        start = i
        continue
    if flag:
        if line.startswith("}"):
            end = i
            break

print(start, end)

response = requests.get("https://api.github.com/repos/xiaosu-zhu/McQuic/releases/tags/generic", headers={"Accept":"application/vnd.github.v3+json"}).json()
assets = response["assets"]

assets = [{
    "name": "qp_1_msssim_12345678.mcquic",
},{
    "name": "qp_2_msssim_12343678.mcquic",
},{
    "name": "qp_3_msssim_123a5678.mcquic",
},{
    "name": "qp_4_msssim_123e5678.mcquic",
},]


MODELS_HASH = dict()

for asset in assets:
    name = asset["name"]
    stem = name.split(".")[0]
    component = stem.split("_")
    qp = component[1]
    hashStr = component[-1]
    print(qp, hashStr)
    if len(hashStr) == 8:
        try:
            int(hashStr, 16)
        except ValueError:
            continue
        MODELS_HASH[int(qp)] = hashStr

MODELS_HASH = """MODELS_HASH = {
%s
}""" % (os.linesep.join(f"    {key}: \"{value}\"" for key, value in MODELS_HASH.items()))

result = lines[:start] + [MODELS_HASH] + lines[(end+1):]

with open(sys.argv[1], "w") as fp:
    fp.writelines(result)
