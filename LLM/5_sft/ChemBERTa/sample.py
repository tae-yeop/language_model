import deepchem
from rdkit import Chem
import sys
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline, RobertaModel, RobertaTokenizer

import torch
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import rdFMCS
from matplotlib import colors
from rdkit.Chem import Draw
from rdkit.Chem.Draw import MolToImage
from PIL import Image


def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol

def find_matches_one(mol, submol):
    #find all matching atoms for each submol in submol_list in mol.
    match_dict = {}
    mols = [mol,submol] #pairwise search
    res=rdFMCS.FindMCS(mols) #,ringMatchesRingOnly=True)
    mcsp = Chem.MolFromSmarts(res.smartsString)
    matches = mol.GetSubstructMatches(mcsp)
    return matches

def get_image(mol, atomset):
    hcolor = colors.to_rgb('green')
    if atomset is not None:
        # highlight the atoms set while drawing the whole molecule.
        img = MolToImage(mol, size=(600, 600),fitImage=True, highlightAtoms=atomset,highlightColor=hcolor)
    else:
        img = MolToImage(mol, size=(400, 400),fitImage=True)
    
    return img

if __name__ == '__main__':
    model = AutoModelForMaskedLM.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
    tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")

    fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)

    smiles_mask = "C1=CC=CC<mask>C1"
    smiles = "C1=CC=CC=C1"

    masked_smi = fill_mask(smiles_mask)

    for smi in masked_smi:
        print(smi)

    # SMILES 문자열 중 한 원자를 [MASK] 처리
    sequence = f"C1=CC=CC={tokenizer.mask_token}1" # "C1=CC=CC=<mask>1"
    substructure = "CC=CC"
    image_list = []

    input = tokenizer.encode(sequence, return_tensors="pt") # SMILES를 BPE tokenizer로 토큰화 후 tensor로 변환
    mask_token_index = torch.where(input == tokenizer.mask_token_id)[1] # 마스크 위치 인덱스를 찾아 나중에 logit 추출할 위치로 사용

    token_logits = model(input)[0]
    mask_token_logits = token_logits[0, mask_token_index, :] # 모델의 output logits에서 [MASK] 토큰 위치에 해당하는 logit 추출

    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist() # Top 5 확률 높은 토큰을 선택

    for token in top_5_tokens:
        smi = (sequence.replace(tokenizer.mask_token, tokenizer.decode([token]))) # [MASK]를 각 예측 토큰으로 바꾸어 SMILES 완성
        print (smi)
        smi_mol = get_mol(smi)
        substructure_mol = get_mol(substructure)
        if smi_mol is None: # if the model's token prediction isn't chemically feasible
            continue

        Draw.MolToFile(smi_mol, smi+".png")
        matches = find_matches_one(smi_mol, substructure_mol)
        atomset = list(matches[0])
        img = get_image(smi_mol, atomset)
        img.format="PNG" 
        image_list.append(img)