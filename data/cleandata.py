import string
data = open("marios_run.txt", "r")
cdata = open("marios_run_clean.txt", "a")

punctuation = "=:_" + string.ascii_lowercase

def remove_punctuation(s):
  return s.translate(str.maketrans("","", punctuation))

for line in data:
  line = remove_punctuation(line)
  cdata.write(line)
