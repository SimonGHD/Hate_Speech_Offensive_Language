# read requirements.txt file, create list of package names

from subprocess import call

f = open("C:/Users/simon/PycharmProjects/Hate_Speech&Offensive_Language/requirements.txt", "r")
if f.mode == 'r':
    requirements = f.readlines()

print(requirements)
for package in requirements:
    call("pip install " + package, shell=True)
