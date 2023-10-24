STAR --runThreadN 10 --runMode genomeGenerate --genomeDir ./STAR_idx --genomeFastaFiles ./GRCh38.primary_assembly.genome.fa --sjdbGTFfile ./gencode.v39.primary_assembly.annotation.gtf --sjdbOverhang 100

STAR --runThreadN 10 --runMode genomeGenerate --genomeDir ./rep_idx --genomeFastaFiles ./Repbase.fasta --limitGenomeGenerateRAM 34261172490

python /picb/rnasys2/huyue/program/clipperhelper-0.1.1/util/make_custom_species_files.py --gtf gencode.v39.primary_assembly.annotation.gtf --species gencode_v39


gawk 'BEGIN{OFS="\t"}$3=="gene"&&$0~"protein_coding"{print $1,$4,$5,$14,"0",$7}' gencode.v39.primary_assembly.annotation.gtf | tr -d '";' > gencode_gene.bed
gawk 'BEGIN{OFS="\t"}$3=="UTR"&&$0~"protein_coding"{coord=$1"_"$4"_"$5;if(!(coord in name)){name[coord]=$16;strand[coord]=$7}}END{for(c in name){split(c,a,"_");print a[1],a[2],a[3],name[c],"0",strand[c]}}' gencode.v39.primary_assembly.annotation.gtf | tr -d '";' > gencode_UTR.bed
awk 'BEGIN{while(getline < "gencode_gene.bed"){start[$4]=$2;end[$4]=$3}OFS="\t"}{d1=$2-start[$4];d2=end[$4]-$2;if($6=="-"){if(d1<d2){r="UTR3"}else{r="UTR5"}}else{if(d1<d2){r="UTR5"}else{r="UTR3"}}print $1,$2,$3,$4"_"r,$5,$6}' gencode_UTR.bed > gencode_UTR_anno.bed
gawk 'BEGIN{OFS="\t"}$3=="CDS"&&$0~"protein_coding"{coord=$1"_"$4"_"$5;if(!(coord in name)){name[coord]=$16;strand[coord]=$7}}END{for(c in name){split(c,a,"_");print a[1],a[2],a[3],name[c]"_CDS","0",strand[c]}}' gencode.v39.primary_assembly.annotation.gtf | tr -d '";' > gencode_CDS.bed
gawk 'BEGIN{OFS="\t"}$3=="exon"&&$0~"protein_coding"{coord=$1"_"$4"_"$5;if(!(coord in name)){name[coord]=$16;strand[coord]=$7}}END{for(c in name){split(c,a,"_");print a[1],a[2],a[3],name[c],"0",strand[c]}}' gencode.v39.primary_assembly.annotation.gtf | tr -d '";' > gencode_exon.bed

##another version
##download primary isoform from appris
gawk 'BEGIN{OFS="\t"}$3=="exon"{print $1,$4,$5,$20,"0",$7}' gencode.v39.primary_assembly.annotation.gtf | tr -d '";' > gencode_exon.bed
gawk 'BEGIN{OFS="\t"}$3=="UTR"&&$0~"protein_coding"{print $1,$4,$5,$20,"0",$7}' gencode.v39.primary_assembly.annotation.gtf | tr -d '";' > gencode_UTR.bed
awk 'BEGIN{while(getline < "gencode_gene.bed"){start[$4]=$2;end[$4]=$3}OFS="\t"}{d1=$2-start[$4];d2=end[$4]-$2;if($6=="-"){if(d1<d2){r="UTR3"}else{r="UTR5"}}else{if(d1<d2){r="UTR5"}else{r="UTR3"}}print $1,$2,$3,$4"_"r,$5,$6}' gencode_UTR.bed > gencode_UTR_anno.bed
gawk 'BEGIN{OFS="\t"}$3=="CDS"&&$0~"protein_coding"{print $1,$4,$5,$20"_CDS","0",$7}' gencode.v39.primary_assembly.annotation.gtf | tr -d '";' > gencode_CDS.bed


gawk 'BEGIN{OFS="\t"}$3=="transcript"&&$0~"protein_coding"{print $1,$4,$5,$20,"0",$7}' gencode.v39.primary_assembly.annotation.gtf | tr -d '";' > gencode_transcript.bed

cat gencode.v39.primary_assembly.annotation.gtf  |grep chr|awk -v FS='\t' '$3=="exon" { exonName=$1":"$4":"$5":"$7; split($9, fields, ";"); geneName=fields[1]; transcriptName=fields[6];gsub("transcript_name ","",transcriptName);printf("%s\t%s\t%s\n",exonName,geneName,transcriptName); }' | tr -d '"' | sort | uniq | awk -v FS='\t' '{ eCount[$1]++; tCount[$3]++; exonHost[$1]=$3; if(tCount[$3]==1) gCount[$2]++; } END { for(i in eCount) { split(i,fields,":"); printf("%s\t%s\t%s\t%s\t.\t%s\t\n",fields[1],fields[2],fields[3],exonHost[i],fields[4]); } }' |  bedtools sort -i stdin |awk -v FS='\t' '{ if( last_exon[$4]==1 && (last_exon_end[$4]+1)<($2-1) ) printf("%s\t%i\t%i\t%s\t%s\t%s\n",$1,last_exon_end[$4]+1,$2-1,$4,$5,$6); last_exon[$4]=1; last_exon_end[$4]=$3; }' > intron.bed

python fixed_length.py -f non_protein_coding_transcripts.fa -t non -r > non_protein_coding_transcripts_random_100.txt
python fixed_length.py -f non_protein_coding_transcripts.fa -t non > non_protein_coding_transcripts_100.txt
python fixed_length.py -f protein_coding_transcripts.fa -s protein_coding_transcripts_regions.txt -r > protein_coding_transcripts_random_100.txt
python fixed_length.py -f protein_coding_transcripts.fa -s protein_coding_transcripts_regions.txt > protein_coding_transcripts_100.txt

cat non_protein_coding_transcripts_100.txt protein_coding_transcripts_100.txt | awk -v OFS="\t" '{print $0,1}' > transcripts_class.txt

cat non_protein_coding_transcripts_random_100.txt protein_coding_transcripts_random_100.txt | awk -v OFS="\t" '{print $0,0}' > non_transcripts_class.txt

cat transcripts_class.txt non_transcripts_class.txt > bert_pretrain.txt