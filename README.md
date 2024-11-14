# PFP-GMB
Protein function prediction using graph neural network with multi-type biological knowledge

## Requirements
python==3.8.16   
numpy==1.19.5     
scipy==1.5.0   
networkx==3.1   
torch==1.12.0+cu102   
dgl==1.1.1+cu102   
click==7.1.2   
ruamel.yaml==0.16.6   
tqdm==4.47.0   
logzero==1.5.0   
joblib==0.16.0 

## Data And Model Link
https://drive.google.com/drive/folders/1FT5aQa_XOvjPt4t-Iv8VBYNocBfpQHaq?usp=drive_link

## For Test
### Data Format
#### {class_tag}_test2.fasta:
>5NTC_RAT
MMTSWSDRLQNAADVPANMDKHALKKYRREAYHRVFVNRSLAMEKIKCFGFDMDYTLAVYKSPEYESLGFELTVERLVSIGYPQELLNFAYDSTFPTRGLVFDTLYGNLLKVDAYGNLLVCAHGFNFIRGPETREQYPNKFIQRDDTERFYILNTLFNLPETYLLACLVDFFTNCPRYTSCDTGFKDGDLFMSYRSMFQDVRDAVDWVHYKGSLKEKTVENLEKYVVKDGKLPLLLSRMKEVGKVFLATNSDYKYTDKIMTYLFDFPHGPKPGSSHRPWQSYFDLILVDARKPLFFGEGTVLRQVDTKTGKLKIGTYTGPLQHGIVYSGGSSDTICDLLGAKGKDILYIGDHIFGDILKSKKRQGWRTFLVIPELAQELHVWTDKSSLFEELQSLDIFLAELYKHLDSSSNERPDISSIQRRIKKVTHDMDMCYGMMGSLFRSGSRQTLFASQVMRYADLYAASFINLLYYPFSYLFRAAHVLMPHESTVEHTHVDINEMESPLATRNRTSVDFKDTDYKRHQLTRSISEIKPPNLFPLAPQEITHCHDEDDDEEEEEEE
>protein_id
sequences

#### {class_tag}_test2_pid_list.txt(存放所有的protein_id):
5NTC_RAT
6PGL_SCHPO
protein_id

#### test2_data_sequence_embeddings.pkl:
{'5NTC_RAT': tensor([-0.1487,  0.1495, -0.0301,  ..., -0.1990, -0.2109, -0.2146])}
{protein_id1:sequence_embedding1,protein_id2:sequence_embedding2,...}

### Modify File
{class_tag}.yaml

### Run Command
python main_multi_test.py -d '{class_tag}' -n {gpu_number}

class_tag:{'bp','cc','mf'}
gpu_number:{0,1,2,3}
