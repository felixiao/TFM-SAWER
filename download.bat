rsync -avPuzh -e ssh --compress-level=9 -o -g -p \
    --temp-dir=/root/autodl-tmp/temp/ --partial-dir=/root/autodl-tmp/temp/ --delay-updates \
    --exclude={'.git','Data','English-Jar','FMLPETER','Result','FMLP','P5','FMLPETER','PEPLER','PETER','Att2Seq','NETE','NRT','__pycache__','*/__pycache__','.DS_Store','*/.DS_Store','*.err','SEQUER/checkpoints'} \
    --itemize-changes \
    nct01141@dt01.bsc.es:/home/nct01/nct01141/torch/ ./