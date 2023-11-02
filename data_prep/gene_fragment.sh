bedtools getfasta -fi GRCh38.primary_assembly.genome.fa -bed gencode_gene_gid.bed -name > gencode_gene_gid.fa
awk 'NR%2==1{gsub(">","",$0);name=$0}NR%2==0{print name"\t"$0}' gencode_gene_gid.fa | /picb/rnasys2/huyue/program/bin/parallel --pipe -q awk '{name=$1;split(name,nlist,":");gid=nlist[1];chr=nlist[3];split(nlist[4],coord,"(");gsub(")","",coord[2]);strand=coord[2];split(coord[1],pos,"-");start=pos[1];end=pos[2];for(i=1;i<=length($2)-50;i=i+50){if(strand=="+"){s=start+i-1;e=start+i+100-2}else{s=end-i-100+2;e=end-i+1}if(i+100<=length($2)){print substr($2,i,100),gid":"chr":"s":"e":"strand}else{if(strand=="+"){s=end-100;e=end-1}else{s=start+1;e=start+100}print substr($2,length($2)-99,100),gid":"chr":"s":"e":"strand}}}' > gene_fragment_100.txt


cat gene_fragment_100.txt | awk -v OFS="\t" '{if(!($1 in d)){d[$1]=$2}else{d[$1]=d[$1]","$2}}END{for(s in d){print s,d[s]}}' > gene_fragment_100_uniq.txt


awk 'NR%2==1{gsub(">","",$0);name=$0}NR%2==0{print name"\t"$0}' transcripts.fa | /picb/rnasys2/huyue/program/bin/parallel --pipe -q awk '{name=$1;split(name,nlist,"|");tid=nlist[1];gid=nlist[2];for(i=1;i<=length($2)-50;i=i+50){s=start+i-1;e=start+i+100-2;if(i+100<=length($2)){print substr($2,i,100),tid":"s":"e}else{s=end-100;e=end-1;print substr($2,length($2)-99,100),tid":"s":"e}}}' > trans_frag_100.txt

cat trans_frag_100.txt | awk -v OFS="\t" '{if(!($1 in d)){d[$1]=$2}else{d[$1]=d[$1]","$2}}END{for(s in d){print s,d[s]}}' > trans_frag_100_uniq.txt