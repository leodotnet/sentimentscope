import subprocess
import sys
import os

def createDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

expDir = 'experiments//sentiment//model//'
models = ['sentimentspan_nonlatent', 'sentimentspan_latent', 'semimarkov_nonlatent', 'semimarkov_latent', 'baseline_collapse', 'baseline_pipeline'£¬'baseline_joint']
langs = ['en', 'es']

createDir('experiments')
createDir('experiments//sentiment')
createDir('experiments//sentiment//model')

for model in models:
    createDir(expDir + model)
    for lang in langs:
        createDir(expDir + model + '//Twitter_' + lang)
        createDir(expDir + model + '//Twitter_' + lang + '//temp')

