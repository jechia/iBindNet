#download from ENCODE database(skipped)
#a.count unique reads from each library
#input: fastq file
#output: unique read count from each library
while FS=" " read -r -a line;
do
 echo "awk 'NR%4==2' ${line[0]}.fastq | sort | uniq -c > ${line[0]}_count.txt";
 awk 'NR%4==2' ${line[0]}.fastq | sort | uniq -c > ${line[0]}_count.txt;
done<down.list

#b0.combine bound and unbound set
#input: pulldown/ no_target&input library
#output: bound/ control set
cat ${rbp}_no_target_count.txt ${rbp}_input_count.txt | awk '{if(!($2 in count)){count[$2]=$1}else{count[$2]=count[$2]+$1}}END{for(s in count){print count[s],s}}' > ${rbp}_control_count.txt
cat ${rbp}_*nM_count.txt | awk '!($2 in d){d[$2]=1}END{for(s in d){print s,d[s]}}' > bound_set.txt

#b.count kmer(k=10) from each library
#input: unique read count
#output: kmer count
awk '{for(i=1;i<=length($1)-9;i=i+5){print substr($1,i,10)}}' bound_set.txt | awk '{if($1 in count){count[$1]=count[$1]+1}else{count[$1]=1}}END{for(k in count){print k,count[k]}}' > kmer_count.txt
sort -k2n kmer_count.txt > kmer_count_sort.txt
awk '{for(i=1;i<=length($2)-9;i=i+5){print substr($2,i,10)}}' ${rbp}_control_count.txt |awk '{if($1 in count){count[$1]=count[$1]+1}else{count[$1]=1}}END{for(k in count){print k,count[k]}}' > ${rbp}_control_kmer_count.txt

#c.normalize kmer based on control set
#input: kmer count from bound/ control set
#output: normalized kmer ratio
##note: change the filename in awk command
awk 'BEGIN{while(getline < "FUS_control_kmer_count.txt"){d[$1]=$2}}$1 in d{print $1,$2/d[$1]}' kmer_count_sort.txt | sort -k2n > kmer_norm_ratio.txt

#d0.calculate the top 0.5% kmer based on normalized kmer ratio
grep -v N kmer_norm_ratio.txt | tail -5246 > kmer_sel.txt
#d.get sequence based on selected kmer
awk 'BEGIN{while(getline< "kmer_sel.txt"){d[$1]=$1}}{for(i=1;i<=length($1)-9;i=i+5){a=substr($1,i,10);if(a in d){print $1;break}}}' bound_set.txt > bound_set_sel.txt
awk 'BEGIN{while(getline < "bound_set.txt"){d[$1]=$1}}!($2 in d)' ${rbp}_control_count.txt > unbound_set.txt

#e.get randome in vitro library
#input: bound, unbound class
#output: total library
sort -R ${rbp}_unbound_class.txt | head -1000000 | awk -v OFS="\t" '{print $1,0}' > ${rbp}_unbound_bert_pretrain.txt
sort -R bound_set_sel.txt | head -1000000 | awk -v OFS="\t" '{print $1,1}' > ${rbp}_bound_bert_pretrain.txt
