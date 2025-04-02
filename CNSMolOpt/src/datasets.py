import os
import numpy as np
import pandas as pd
import pickle
import torch

from rdkit import Chem
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src import const

def read_sdf(sdf_path):
    with Chem.SDMolSupplier(sdf_path, sanitize=False) as supplier:
        for molecule in supplier:
            yield molecule

def get_one_hot(atom, atoms_dict):
    one_hot = np.zeros(len(atoms_dict))
    one_hot[atoms_dict[atom]] = 1
    return one_hot

def parse_molecule(mol):
    one_hot = []
    charges = []
    atom2idx = const.ATOM2IDX
    charges_dict = const.CHARGES
    for atom in mol.GetAtoms():
        one_hot.append(get_one_hot(atom.GetSymbol(), atom2idx))
        charges.append(charges_dict[atom.GetSymbol()])
    positions = mol.GetConformer().GetPositions()
    return positions, np.array(one_hot), np.array(charges)

# for multi given size
def parse_rgroup_multian(mol):
    mol_frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    one_hot_list = []
    charges_list = []
    positions_list = []
    rgroup_atom_num_list = []
    atom2idx = const.ATOM2IDX
    charges_dict = const.CHARGES
    for i in range(len(mol_frags)):
        one_hot = []
        charges = []
        for atom in mol_frags[i].GetAtoms():
            one_hot.append(get_one_hot(atom.GetSymbol(), atom2idx))
            charges.append(charges_dict[atom.GetSymbol()])
        positions = mol_frags[i].GetConformer().GetPositions()
        positions = positions.tolist()
        one_hot_list.append(one_hot)
        charges_list.append(charges)
        positions_list.append(positions)
        rgroup_atom_num_list.append(mol_frags[i].GetNumAtoms())
        # rgroup_atom_num_list.append(10)
    one_hot = sum(one_hot_list, [])
    charges = sum(charges_list, [])
    positions = sum(positions_list, [])
    return np.array(positions), np.array(one_hot), np.array(charges), rgroup_atom_num_list

# for multi w/o anchors, with fake atom
def parse_rgroup_com_scaf(mol, fake_pos):
    mol_frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    one_hot_list = []
    charges_list = []
    positions_list = []
    atom2idx = const.ATOM2IDX
    charges_dict = const.CHARGES
    for i in range(len(mol_frags)):
        one_hot = []
        charges = []
        for atom in mol_frags[i].GetAtoms():
            one_hot.append(get_one_hot(atom.GetSymbol(), atom2idx))
            charges.append(charges_dict[atom.GetSymbol()])
        positions = mol_frags[i].GetConformer().GetPositions()
        one_hot.extend(get_one_hot('#', atom2idx) for _ in range(10 - mol_frags[i].GetNumAtoms()))
        charges.extend(charges_dict['#'] for _ in range(10 - mol_frags[i].GetNumAtoms()))
        positions = positions.tolist()
        positions.extend(fake_pos for _ in range(10 - mol_frags[i].GetNumAtoms()))
        one_hot_list.append(one_hot)
        charges_list.append(charges)
        positions_list.append(positions)
    one_hot = sum(one_hot_list, [])
    charges = sum(charges_list, [])
    positions = sum(positions_list, [])
    return np.array(positions), np.array(one_hot), np.array(charges)

# for single or multi with anchors, with fake atom
def parse_rgroup(mol, fake_pos):
    if isinstance(fake_pos[0], list):
        mol_frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        assert len(mol_frags) == len(fake_pos)
        one_hot_list = []
        charges_list = []
        positions_list = []
        atom2idx = const.ATOM2IDX
        charges_dict = const.CHARGES
        for i in range(len(mol_frags)):
            one_hot = []
            charges = []
            for atom in mol_frags[i].GetAtoms():
                one_hot.append(get_one_hot(atom.GetSymbol(), atom2idx))
                charges.append(charges_dict[atom.GetSymbol()])
            positions = mol_frags[i].GetConformer().GetPositions()
            one_hot.extend(get_one_hot('#', atom2idx) for _ in range(10 - mol_frags[i].GetNumAtoms()))
            charges.extend(charges_dict['#'] for _ in range(10 - mol_frags[i].GetNumAtoms()))
            positions = positions.tolist()
            positions.extend(fake_pos[i] for _ in range(10 - mol_frags[i].GetNumAtoms()))
            one_hot_list.append(one_hot)
            charges_list.append(charges)
            positions_list.append(positions)
        one_hot = sum(one_hot_list, [])
        charges = sum(charges_list, [])
        positions = sum(positions_list, [])
    else:
        fake_pos = list(fake_pos)
        one_hot = []
        charges = []
        atom2idx = const.ATOM2IDX
        charges_dict = const.CHARGES
        for atom in mol.GetAtoms():
            one_hot.append(get_one_hot(atom.GetSymbol(), atom2idx))
            charges.append(charges_dict[atom.GetSymbol()])
        positions = mol.GetConformer().GetPositions()
        one_hot.extend(get_one_hot('#', atom2idx) for _ in range(10 - mol.GetNumAtoms()))
        charges.extend(charges_dict['#'] for _ in range(10 - mol.GetNumAtoms()))
        positions = positions.tolist()
        positions.extend(fake_pos for _ in range(10 - mol.GetNumAtoms()))
    return np.array(positions), np.array(one_hot), np.array(charges)

def collate(batch):
    out = {}

    for i, data in enumerate(batch):
        for key, value in data.items():
            out.setdefault(key, []).append(value)

    for key, value in out.items():
        if key in const.DATA_LIST_ATTRS:
            continue
        if key in const.DATA_ATTRS_TO_PAD:
            out[key] = torch.nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=0)
            continue
        raise Exception(f'Unknown batch key: {key}')

    atom_mask = (out['scaffold_mask'].bool() | out['rgroup_mask'].bool()).to(const.TORCH_INT)
    out['atom_mask'] = atom_mask[:, :, None]

    batch_size, n_nodes = atom_mask.size()

    if 'pocket_mask' in batch[0].keys():
        batch_mask = torch.cat([
            torch.ones(n_nodes, dtype=const.TORCH_INT) * i
            for i in range(batch_size)
        ]).to(atom_mask.device)
        out['edge_mask'] = batch_mask
    else:
        edge_mask = atom_mask[:, None, :] * atom_mask[:, :, None]
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=const.TORCH_INT, device=atom_mask.device).unsqueeze(0)
        edge_mask *= diag_mask
        out['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

    for key in const.DATA_ATTRS_TO_ADD_LAST_DIM:
        if key in out.keys():
            out[key] = out[key][:, :, None]

    return out

def collate_with_scaffold_edges(batch):
    out = {}

    for i, data in enumerate(batch):
        for key, value in data.items():
            out.setdefault(key, []).append(value)

    for key, value in out.items():
        if key in const.DATA_LIST_ATTRS:
            continue
        if key in const.DATA_ATTRS_TO_PAD:
            out[key] = torch.nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=0)
            continue
        raise Exception(f'Unknown batch key: {key}')

    scaf_mask = out['scaffold_mask']
    # scaf_mask = out['scaffold_only_mask']
    edge_mask = scaf_mask[:, None, :] * scaf_mask[:, :, None]
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=const.TORCH_INT, device=scaf_mask.device).unsqueeze(0)
    edge_mask *= diag_mask

    batch_size, n_nodes = scaf_mask.size()
    out['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

    # Building edges and covalent bond values
    rows, cols, bonds = [], [], []
    for batch_idx in range(batch_size):
        for i in range(n_nodes):
            for j in range(n_nodes):
                rows.append(i + batch_idx * n_nodes)
                cols.append(j + batch_idx * n_nodes)

    edges = [torch.LongTensor(rows).to(scaf_mask.device), torch.LongTensor(cols).to(scaf_mask.device)]
    out['edges'] = edges

    atom_mask = (out['scaffold_mask'].bool() | out['rgroup_mask'].bool()).to(const.TORCH_INT)
    # atom_mask = (out['scaffold_only_mask'].bool() | out['rgroup_mask'].bool()).to(const.TORCH_INT)
    out['atom_mask'] = atom_mask[:, :, None]

    for key in const.DATA_ATTRS_TO_ADD_LAST_DIM:
        if key in out.keys():
            out[key] = out[key][:, :, None]

    return out

def get_dataloader(dataset, batch_size, collate_fn=collate, shuffle=False):
    return DataLoader(dataset, batch_size, collate_fn=collate_fn, shuffle=shuffle)

def create_template(tensor, scaffold_size, rgroup_size, fill=0):
    values_to_keep = tensor[:scaffold_size]
    values_to_add = torch.ones(rgroup_size, tensor.shape[1], dtype=values_to_keep.dtype, device=values_to_keep.device)
    values_to_add = values_to_add * fill
    return torch.cat([values_to_keep, values_to_add], dim=0)

def create_templates_for_rgroup_generation_single(data, rgroup_sizes):#解释
    decoupled_data = []
    for i, rgroup_size in enumerate(rgroup_sizes):
        data_dict = {}
        scaffold_mask = data['scaffold_mask'][i].squeeze()
        scaffold_size = scaffold_mask.sum().int()
        for k, v in data.items():
            if k == 'num_atoms':
                data_dict[k] = scaffold_size + rgroup_size
                continue
            if k in const.DATA_LIST_ATTRS:
                data_dict[k] = v[i]
                continue
            if k in const.DATA_ATTRS_TO_PAD:
                fill_value = 1 if k == 'rgroup_mask' else 0
                template = create_template(v[i], scaffold_size, rgroup_size, fill=fill_value)
                if k in const.DATA_ATTRS_TO_ADD_LAST_DIM:
                    template = template.squeeze(-1)
                data_dict[k] = template

        decoupled_data.append(data_dict)

    return collate(decoupled_data) # for single

def create_templates_for_rgroup_generation_multi(data, rgroup_sizes):#这里
    #print(data.keys())
    decoupled_data = []
    for i, rgroup_size in enumerate(rgroup_sizes):
        data_dict = {}
        scaffold_mask = data['scaffold_mask'][i].squeeze()
        scaffold_size = scaffold_mask.sum().int()
        for k, v in data.items():
            if k == 'num_atoms':
                data_dict[k] = scaffold_size + rgroup_size 
                continue
           
            if k in const.DATA_LIST_ATTRS or k in const.DATA_ATTRS_TO_aff:
                data_dict[k] = v[i]
                continue
            if k in const.DATA_ATTRS_TO_PAD:

                # 检查 v[i] 的维度
                if len(v[i].shape) == 1:
                    print(f"Warning: Key '{k}' has a 1D array for molecule {i}: {v[i]}")
                fill_value = 1 if k == 'rgroup_mask' else 0
                try:
                    template = create_template(v[i], scaffold_size, rgroup_size, fill=fill_value)
                    if k in const.DATA_ATTRS_TO_ADD_LAST_DIM:
                        template = template.squeeze(-1)
                    data_dict[k] = template
                except IndexError as e:
                    print(f"Error processing key '{k}' with value '{v[i]}' for molecule {i}")
                    raise e

        decoupled_data.append(data_dict)

    return collate_mr(decoupled_data) # for multi

# single
class CrossDockDataset(Dataset):
    def __init__(self, data_path, prefix, device):
        if '.' in prefix:
            prefix, pocket_mode = prefix.split('.')
        else:
            parts = prefix.split('_')
            prefix = '_'.join(parts[:-1])
            pocket_mode = parts[-1]

        dataset_path = os.path.join(data_path, f'{prefix}_{pocket_mode}.pt')
        if os.path.exists(dataset_path):
            self.data = torch.load(dataset_path, map_location=device)
        else:
            print(f'Preprocessing dataset with prefix {prefix}')
            self.data = CrossDockDataset.preprocess(data_path, prefix, pocket_mode, device)
            torch.save(self.data, dataset_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @staticmethod
    def preprocess(data_path, prefix, pocket_mode, device):#pocket_mode是什么
        data = []
        table_path = os.path.join(data_path, f'{prefix}_table.csv')
        scaffold_path = os.path.join(data_path, f'{prefix}_scaf.sdf')
        rgroups_path = os.path.join(data_path, f'{prefix}_rgroup.sdf')
        pockets_path = os.path.join(data_path, f'{prefix}_pockets.pkl')

        with open(pockets_path, 'rb') as f:
            pockets = pickle.load(f)

        table = pd.read_csv(table_path)
        generator = tqdm(
            zip(table.iterrows(), read_sdf(scaffold_path), read_sdf(rgroups_path), pockets),
            total=len(table)
        )
        for (_, row), scaffold, rgroup, pocket_data in generator:
            if type(scaffold) != Chem.rdchem.Mol or type(rgroup) != Chem.rdchem.Mol:
                continue
            uuid = row['uuid']
            # cat = row['cat']
            name = row['molecule']
            anchor_id = row['anchor']
            protein_filename = row['protein_filename']
            scaf_pos, scaf_one_hot, scaf_charges = parse_molecule(scaffold)
            
            # fake_pos = np.mean(scaf_pos, axis = 0) # fake atom of scaf
            fake_pos = scaf_pos[anchor_id] # fake atom of anchor
            
            rgroup_pos, rgroup_one_hot, rgroup_charges = parse_rgroup(rgroup, fake_pos)#假原子
            # rgroup_pos, rgroup_one_hot, rgroup_charges = parse_molecule(rgroup)

            pocket_pos = []#靶点的
            pocket_one_hot = []
            pocket_charges = []
            for i in range(len(pocket_data[f'{pocket_mode}_types'])):
                atom_type = pocket_data[f'{pocket_mode}_types'][i]
                pos = pocket_data[f'{pocket_mode}_coord'][i]
                if atom_type == 'H':
                    continue
                pocket_pos.append(pos)
                pocket_one_hot.append(get_one_hot(atom_type, const.ATOM2IDX))#类型转换成数组
                pocket_charges.append(const.CHARGES[atom_type])
            pocket_one_hot = np.array(pocket_one_hot)
            pocket_charges = np.array(pocket_charges)
            pocket_pos = np.array(pocket_pos)

            positions = np.concatenate([scaf_pos, pocket_pos, rgroup_pos], axis=0)
            one_hot = np.concatenate([scaf_one_hot, pocket_one_hot, rgroup_one_hot], axis=0)
            charges = np.concatenate([scaf_charges, pocket_charges, rgroup_charges], axis=0)
            anchors = np.zeros_like(charges)

            anchors[row['anchor']] = 1

            scaf_only_mask = np.concatenate([
                np.ones_like(scaf_charges),
                np.zeros_like(pocket_charges),
                np.zeros_like(rgroup_charges)
            ])
            pocket_mask = np.concatenate([
                np.zeros_like(scaf_charges),
                np.ones_like(pocket_charges),
                np.zeros_like(rgroup_charges)
            ])
            rgroup_mask = np.concatenate([
                np.zeros_like(scaf_charges),
                np.zeros_like(pocket_charges),
                np.ones_like(rgroup_charges)
            ])
            scaf_mask = np.concatenate([
                np.ones_like(scaf_charges),
                np.ones_like(pocket_charges),
                np.zeros_like(rgroup_charges)
            ])

            data.append({
                'uuid': uuid,
                'name': name,
                'positions': torch.tensor(positions, dtype=const.TORCH_FLOAT, device=device),
                'one_hot': torch.tensor(one_hot, dtype=const.TORCH_FLOAT, device=device),
                'charges': torch.tensor(charges, dtype=const.TORCH_FLOAT, device=device),
                'anchors': torch.tensor(anchors, dtype=const.TORCH_FLOAT, device=device),
                'scaffold_only_mask': torch.tensor(scaf_only_mask, dtype=const.TORCH_FLOAT, device=device),
                'pocket_mask': torch.tensor(pocket_mask, dtype=const.TORCH_FLOAT, device=device),
                'scaffold_mask': torch.tensor(scaf_mask, dtype=const.TORCH_FLOAT, device=device),
                'rgroup_mask': torch.tensor(rgroup_mask, dtype=const.TORCH_FLOAT, device=device),
                'num_atoms': len(positions),
                # 'cat': cat,
            })

        return data

    @staticmethod
    def create_edges(positions, scaffold_mask_only, rgroup_mask_only):
        ligand_mask = scaffold_mask_only.astype(bool) | rgroup_mask_only.astype(bool)
        ligand_adj = ligand_mask[:, None] & ligand_mask[None, :]
        proximity_adj = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1) <= 6
        full_adj = ligand_adj | proximity_adj
        full_adj &= ~np.eye(len(positions)).astype(bool)

        curr_rows, curr_cols = np.where(full_adj)
        return [curr_rows, curr_cols]

# multi w/o anchor
class MultiRDataset(Dataset):
    def __init__(self, data_path, prefix, device):
        if '.' in prefix:
            prefix, pocket_mode = prefix.split('.')
        else:
            parts = prefix.split('_')
            prefix = '_'.join(parts[:-1])
            pocket_mode = parts[-1]

        dataset_path = os.path.join(data_path, f'{prefix}_{pocket_mode}.pt')
        if os.path.exists(dataset_path):
            self.data = torch.load(dataset_path, map_location=device)
        else:
            print(f'Preprocessing dataset with prefix {prefix}')
            self.data = MultiRDataset.preprocess(data_path, prefix, pocket_mode, device)
            torch.save(self.data, dataset_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @staticmethod
    def preprocess(data_path, prefix, pocket_mode, device):
        data = []
        table_path = os.path.join(data_path, f'{prefix}_table.csv')
        scaffold_path = os.path.join(data_path, f'{prefix}_scaf.sdf')
        rgroups_path = os.path.join(data_path, f'{prefix}_rgroup.sdf')
        pockets_path = os.path.join(data_path, f'{prefix}_pockets.pkl')

        with open(pockets_path, 'rb') as f:
            pockets = pickle.load(f)

        table = pd.read_csv(table_path)
        generator = tqdm(
            zip(table.iterrows(), read_sdf(scaffold_path), read_sdf(rgroups_path), pockets),
            total=len(table)
        )
        for (_, row), scaffold, rgroup, pocket_data in generator:
            if type(scaffold) != Chem.rdchem.Mol or type(rgroup) != Chem.rdchem.Mol:
                continue
            uuid = row['uuid']
            name = row['molecule']
            anchor_id = row['anchor']
            scaf_pos, scaf_one_hot, scaf_charges = parse_molecule(scaffold)
            fake_pos = np.mean(scaf_pos, axis = 0) # fake atom of scaf
            fake_pos = scaf_pos[anchor_id]
            rgroup_pos, rgroup_one_hot, rgroup_charges = parse_rgroup_com_scaf(rgroup, fake_pos)
            # rgroup_pos, rgroup_one_hot, rgroup_charges = parse_molecule(rgroup)

            pocket_pos = []
            pocket_one_hot = []
            pocket_charges = []
            for i in range(len(pocket_data[f'{pocket_mode}_types'])):
                atom_type = pocket_data[f'{pocket_mode}_types'][i]
                pos = pocket_data[f'{pocket_mode}_coord'][i]
                if atom_type == 'H':
                    continue
                pocket_pos.append(pos)
                pocket_one_hot.append(get_one_hot(atom_type, const.ATOM2IDX))
                pocket_charges.append(const.CHARGES[atom_type])
            pocket_one_hot = np.array(pocket_one_hot)
            pocket_charges = np.array(pocket_charges)
            pocket_pos = np.array(pocket_pos)

            positions = np.concatenate([scaf_pos, pocket_pos, rgroup_pos], axis=0)
            one_hot = np.concatenate([scaf_one_hot, pocket_one_hot, rgroup_one_hot], axis=0)
            charges = np.concatenate([scaf_charges, pocket_charges, rgroup_charges], axis=0)
            anchors = np.zeros_like(charges)

            for anchor_idx in map(int, row['anchor'].split('|')):
                anchors[anchor_idx] = 1

            scaf_only_mask = np.concatenate([
                np.ones_like(scaf_charges),
                np.zeros_like(pocket_charges),
                np.zeros_like(rgroup_charges)
            ])
            pocket_mask = np.concatenate([
                np.zeros_like(scaf_charges),
                np.ones_like(pocket_charges),
                np.zeros_like(rgroup_charges)
            ])
            rgroup_mask = np.concatenate([
                np.zeros_like(scaf_charges),
                np.zeros_like(pocket_charges),
                np.ones_like(rgroup_charges)
            ])
            scaf_mask = np.concatenate([
                np.ones_like(scaf_charges),
                np.ones_like(pocket_charges),
                np.zeros_like(rgroup_charges)
            ])

            data.append({
                'uuid': uuid,
                'name': name,
                'positions': torch.tensor(positions, dtype=const.TORCH_FLOAT, device=device),
                'one_hot': torch.tensor(one_hot, dtype=const.TORCH_FLOAT, device=device),
                'charges': torch.tensor(charges, dtype=const.TORCH_FLOAT, device=device),
                'anchors': torch.tensor(anchors, dtype=const.TORCH_FLOAT, device=device),
                'scaffold_only_mask': torch.tensor(scaf_only_mask, dtype=const.TORCH_FLOAT, device=device),
                'pocket_mask': torch.tensor(pocket_mask, dtype=const.TORCH_FLOAT, device=device),
                'scaffold_mask': torch.tensor(scaf_mask, dtype=const.TORCH_FLOAT, device=device),
                'rgroup_mask': torch.tensor(rgroup_mask, dtype=const.TORCH_FLOAT, device=device),
                'num_atoms': len(positions),
            })

        return data

    @staticmethod
    def create_edges(positions, scaffold_mask_only, rgroup_mask_only):
        ligand_mask = scaffold_mask_only.astype(bool) | rgroup_mask_only.astype(bool)
        ligand_adj = ligand_mask[:, None] & ligand_mask[None, :]
        proximity_adj = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1) <= 6
        full_adj = ligand_adj | proximity_adj
        full_adj &= ~np.eye(len(positions)).astype(bool)

        curr_rows, curr_cols = np.where(full_adj)
        return [curr_rows, curr_cols]

def discretize_affinity_two(value, bins):
    # 将亲和力值映射到对应的区间
    if value <= bins[0]:
        return 0  # 低亲和力
    else:
        return 1

def discretize_affinity_three(value, bins):
    # 将亲和力值映射到对应的区间
    if value <= bins[0]:
        return 0  # 低亲和力
    elif value <= bins[1]:
        return 1
    else:
        return 2

def discretize_affinity(value, bins):
    # 将亲和力值映射到对应的区间
    if value <= bins[0]:
        return 0  # 低亲和力
    elif value <= bins[1]:
        return 1  # 中低亲和力
    elif value <= bins[2]:
        return 2  # 中高亲和力
    elif value <= bins[3]:
        return 3
    elif value <= bins[4]:
        return 4
    elif value <= bins[5]:
        return 5
    elif value <= bins[6]:
        return 6
    elif value <= bins[7]:
        return 7
    elif value <= bins[8]:
        return 8
    elif value <= bins[9]:
        return 9
    elif value <= bins[10]:
        return 10
    elif value <= bins[11]:
        return 11
    elif value <= bins[12]:
        return 12
    else:
        return 13  # 高亲和力



# multi with anchor
class MultiRDataset_anchor(Dataset):
    def __init__(self, data_path, prefix, device):
        if '.' in prefix:
            prefix, pocket_mode = prefix.split('.')
        else:
            parts = prefix.split('_')
            prefix = '_'.join(parts[:-1])
            pocket_mode = parts[-1]

        dataset_path = os.path.join(data_path, f'{prefix}_{pocket_mode}.pt')
        if os.path.exists(dataset_path):
            self.data = torch.load(dataset_path, map_location=device)
        else:
            print(f'Preprocessing dataset with prefix {prefix}')
            self.data = MultiRDataset_anchor.preprocess(data_path, prefix, pocket_mode, device)#重头准备
            torch.save(self.data, dataset_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]#加上索引

    @staticmethod
    def preprocess(data_path, prefix, pocket_mode, device):
        data = []
        table_path = os.path.join(data_path, f'{prefix}_table.csv')
        scaffold_path = os.path.join(data_path, f'{prefix}_scaf.sdf')
        rgroups_path = os.path.join(data_path, f'{prefix}_rgroup.sdf')
        pockets_path = os.path.join(data_path, f'{prefix}_pockets.pkl')

        with open(pockets_path, 'rb') as f:
            pockets = pickle.load(f)

        table = pd.read_csv(table_path)
        generator = tqdm(
            zip(table.iterrows(), read_sdf(scaffold_path), read_sdf(rgroups_path), pockets),
            total=len(table)
        )
        for (_, row), scaffold, rgroup, pocket_data in generator:#分别处理
            if type(scaffold) != Chem.rdchem.Mol or type(rgroup) != Chem.rdchem.Mol:
                continue
            #se_code = row['se_code']
            uuid = row['uuid']
            name = row['molecule']
            #anchor_id_list = row['anchor'].split('|')
            if isinstance(row['anchor'], str):
               anchor_id_list = row['anchor'].split('|')
            else:
               anchor_id_list = [str(row['anchor'])]  # 将整数转换为单元素列表
            scaf_pos, scaf_one_hot, scaf_charges = parse_molecule(scaffold)#得到骨架得原子坐标，原子类型，原子电荷
            fake_pos_list = [list(scaf_pos[int(anchor_id)]) for anchor_id in anchor_id_list]#假原子
            try:
                rgroup_pos, rgroup_one_hot, rgroup_charges = parse_rgroup(rgroup, fake_pos_list)
            except Exception as e:
                print(f"Error parsing rgroup for molecule {uuid}: {e}")#
                continue
            # rgroup_pos, rgroup_one_hot, rgroup_charges, rgroup_atom_num_list = parse_rgroup_multian(rgroup)
            rgroup_size_str = '10' # str(rgroup_atom_num_list[0])
            for r_i in range(1, len(anchor_id_list)):
                rgroup_size_str += '|10'# + str(rgroup_atom_num_list[r_i])
            
            pocket_pos = []
            pocket_one_hot = []
            pocket_charges = []
            for i in range(len(pocket_data[f'{pocket_mode}_types'])):
                atom_type = pocket_data[f'{pocket_mode}_types'][i]#在process_and_prepare中
                pos = pocket_data[f'{pocket_mode}_coord'][i]
                if atom_type == 'H':
                    continue
                pocket_pos.append(pos)
                pocket_one_hot.append(get_one_hot(atom_type, const.ATOM2IDX))
                pocket_charges.append(const.CHARGES[atom_type])
            pocket_one_hot = np.array(pocket_one_hot)
            pocket_charges = np.array(pocket_charges)# np.array数组
            pocket_pos = np.array(pocket_pos)

            positions = np.concatenate([scaf_pos, pocket_pos, rgroup_pos], axis=0)
            one_hot = np.concatenate([scaf_one_hot, pocket_one_hot, rgroup_one_hot], axis=0)
            charges = np.concatenate([scaf_charges, pocket_charges, rgroup_charges], axis=0)#沿着1维上拼接
            anchors = np.zeros_like(charges)
            #affinity_matrix = np.zeros_like(charges)

            #bins = [5.872571, 6.637453, 7.173925, 7.697453, 8.119587, 8.69897]#7分类
            #bins = [5.27417001, 5.886056648, 6.292764025, 6.638272164, 6.931814138, 7.178486472, 7.406713933, 7.638272164, 7.886056648, 8.124938737, 8.397940009, 8.698970004, 9.108629825]#14分类
            bins = [8]
            affinity_value = float(row['affinity'])  # 如果是字符串，先转成 float

            # 计算所属区间
            affinity_index = discretize_affinity_two(affinity_value, bins)
            affinity_one_hot = np.zeros(len(bins) + 1) 
            affinity_one_hot[affinity_index] = 1

            # 创建一个全是 0 的矩阵，大小为 4×4（行数固定为区间数+1）
            #affinity_one_hot = np.zeros((len(bins) + 1, len(bins) + 1))  # 4×4 矩阵

            # 在 affinity_index 对应的行中填充 1
            # 假设想让该行是标准 One-Hot 编码，其余行保持为零
            #affinity_one_hot[affinity_index, :] = np.eye(len(bins) + 1)[affinity_index]

            # 将矩阵转换为 Tensor
            affinity_tensor = torch.tensor(affinity_one_hot, dtype=const.TORCH_FLOAT, device=device)

            #affinity_value = row['affinity']  # 提取 affinity 数值（假设为字符串或浮点型）
            
            # Step 2: 初始化与 `charges` 形状一致的矩阵

           
             # 判断 anchor 是整数还是字符串，并进行处理
            if isinstance(row['anchor'], int):
    # 直接将整数作为单一 anchor 处理
               anchor_values = [row['anchor']]
            else:
    # 将字符串拆分为多个整数
               anchor_values = map(int, str(row['anchor']).split('|'))
            #for anchor_idx in map(int, row['anchor'].split('|')):
            for anchor_idx in anchor_values:
                anchors[anchor_idx] = 1
            
            scaf_only_mask = np.concatenate([
                np.ones_like(scaf_charges),
                np.zeros_like(pocket_charges),
                np.zeros_like(rgroup_charges)
            ])#掩码将scaf在矩阵中得位置标记出来，意思就是说哪些位置是scaf
            pocket_mask = np.concatenate([
                np.zeros_like(scaf_charges),
                np.ones_like(pocket_charges),
                np.zeros_like(rgroup_charges)
            ])
            rgroup_mask = np.concatenate([
                np.zeros_like(scaf_charges),
                np.zeros_like(pocket_charges),
                np.ones_like(rgroup_charges)
            ])
            scaf_mask = np.concatenate([
                np.ones_like(scaf_charges),
                np.ones_like(pocket_charges),
                np.zeros_like(rgroup_charges)
            ])#骨架和靶点的位置都标记出来。

            #affinity_value = row['affinity']  # 提取 affinity 数值（假设为字符串）
            #affinity_label = discretize_affinity(affinity_value, bins)
            #affinity_label = float(affinity_label)  # 转为浮点型

            data.append({
                'uuid': uuid,
                'name': name,
                'positions': torch.tensor(positions, dtype=const.TORCH_FLOAT, device=device),
                'one_hot': torch.tensor(one_hot, dtype=const.TORCH_FLOAT, device=device),
                'charges': torch.tensor(charges, dtype=const.TORCH_FLOAT, device=device),
                'anchors': torch.tensor(anchors, dtype=const.TORCH_FLOAT, device=device),
                'scaffold_only_mask': torch.tensor(scaf_only_mask, dtype=const.TORCH_FLOAT, device=device),
                'pocket_mask': torch.tensor(pocket_mask, dtype=const.TORCH_FLOAT, device=device),
                'scaffold_mask': torch.tensor(scaf_mask, dtype=const.TORCH_FLOAT, device=device),
                'rgroup_mask': torch.tensor(rgroup_mask, dtype=const.TORCH_FLOAT, device=device),
                'num_atoms': len(positions),
                'rgroup_size': rgroup_size_str,
                'anchors_str': row['anchor'],
                'affinity_label': torch.tensor(affinity_tensor, dtype=const.TORCH_FLOAT, device=device)
                #'se_code':se_code # 新增字段
            })

        return data

    @staticmethod
    def create_edges(positions, scaffold_mask_only, rgroup_mask_only):
        ligand_mask = scaffold_mask_only.astype(bool) | rgroup_mask_only.astype(bool)
        ligand_adj = ligand_mask[:, None] & ligand_mask[None, :]
        proximity_adj = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1) <= 6
        full_adj = ligand_adj | proximity_adj
        full_adj &= ~np.eye(len(positions)).astype(bool)

        curr_rows, curr_cols = np.where(full_adj)
        return [curr_rows, curr_cols]
    
def collate_mr(batch):
    out = {}
    #print(data.keys())
    #exit()
    for i, data in enumerate(batch):#enumerate得到迭代索引和值
        for key, value in data.items():#数据批次。
            out.setdefault(key, []).append(value)#将值传递给out数组，果字典中不存在该键，则返回这个默认值
    #print(type(out['positions']))
    #exit()
    for key, value in out.items():
        if key in const.DATA_LIST_ATTRS:
            continue
        if key in const.DATA_ATTRS_TO_PAD or const.DATA_ATTRS_TO_aff:
            out[key] = torch.nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=0)#将长度不一致的序列补成一致的。batch_first=True的形状会是(batch_size, max_sequence_length)
            continue
        #if key in const.DATA_ATTRS_TO_aff:
            #out[key] = torch.nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=0)
            #continue
        raise Exception(f'Unknown batch key: {key}')

    atom_mask = (out['scaffold_mask'].bool() | out['rgroup_mask'].bool()).to(const.TORCH_INT)#找出他们都是1的部分，把padding的都标记成0，所有原子的掩码。
    out['atom_mask'] = atom_mask[:, :, None]#在后面增加一个维度

    batch_size, n_nodes = atom_mask.size()

    if 'pocket_mask' in batch[0].keys():#检查batch的第一个样本的键
        batch_mask = torch.cat([
            torch.ones(n_nodes, dtype=const.TORCH_INT) * i
            for i in range(batch_size)
        ]).to(atom_mask.device)#标记每个样本的值属于哪一批次
        out['edge_mask'] = batch_mask
        #print(f"{out['edge_mask'].shape}")

        #exit()
    else:
        edge_mask = atom_mask[:, None, :] * atom_mask[:, :, None]
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=const.TORCH_INT, device=atom_mask.device).unsqueeze(0)
        edge_mask *= diag_mask
        out['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

    for key in const.DATA_ATTRS_TO_ADD_LAST_DIM:
        if key in out.keys():
            out[key] = out[key][:, :, None]

    x = out['positions']
    h = out['one_hot']
    node_mask = out['atom_mask']
    edge_mask = out['edge_mask']#这个是什么
    anchors = out['anchors']
    affinity_label = out['affinity_label']
    scaffold_mask = out['scaffold_mask']
    rgroup_mask = out['rgroup_mask']
    #print(f"context shape: {rgroup_mask.shape}")
    pocket_mask = out['pocket_mask']
    scaffold_only_mask = out['scaffold_only_mask']
    rgroup_size = out['rgroup_size']
    anchors_str = out['anchors_str']
    #print(anchors_str)
    #print(len(anchors_str))

    anchor_id_list = []
    for i in range(len(anchors_str)):
            if isinstance(anchors_str[i], int):
                #tmp = [str(anchors_str[i])]
                anchor_id_list.append(anchors_str[i])
            else:
                tmp = anchors_str[i].split('|') 
                for j in range(len(tmp)):
                    anchor_id_list.append(tmp[j])
    #print(len(anchor_id_list))
    anchors_ = torch.zeros([len(anchor_id_list), anchors.shape[1], anchors.shape[2]]).to(rgroup_mask.device)
    #print(f"anchor_id_list:{anchors_.shape}")
   
    
    for i in range(len(anchor_id_list)):
        anchors_[i][int(anchor_id_list[i])] = 1#在特定的位置添加1
    out['anchors_'] = anchors_

    rgroup_size_list = []
    batch_new_len_list = []
    
    for i in range(len(rgroup_size)):#批次，然后看有多少个
        tmp = rgroup_size[i].split('|')
        tmp_l = []
        for j in range(len(tmp)):
            tmp_l.append(int(tmp[j]))
        rgroup_size_list.append(tmp_l)
        batch_new_len_list.append(len(tmp_l))
    batch_new_len_tensor = torch.tensor(batch_new_len_list).to(rgroup_mask.device)
    #print(batch_new_len_tensor)
   # exit()
    out['batch_new_len_tensor'] = batch_new_len_tensor

    rgroup_idx = torch.nonzero(rgroup_mask)
    # start_idx = [i for i in range(0, anchors_.shape[0] * 10, 10)] # fake atom
    start_idx = [0]
    for i in range(len(rgroup_size_list)):
        for j in range(len(rgroup_size_list[i])):
            start_idx.append(start_idx[-1] + rgroup_size_list[i][j])
    tmp_idx = rgroup_idx[start_idx[:-1]].tolist()
    #print(f"anchors_ shape: {anchors_.shape}")
    #print(f"anchors_ content: {anchors_}")
    rgroup_mask_batch_new = torch.zeros([anchors_.shape[0], rgroup_mask.shape[1], rgroup_mask.shape[2]]).to(rgroup_mask.device)
    cnt = 0
    #print(f"rgroup_size_list: {rgroup_size_list}")
    #print(f"start_idx: {start_idx}")
    #print(f"rgroup_idx shape: {rgroup_idx.shape}")
    #print(f"tmp_idx: {tmp_idx}")
    #print(f"rgroup_mask_batch_new shape: {rgroup_mask_batch_new.shape}")
    #exit()
    for i in range(len(rgroup_size_list)):
        for j in range(len(rgroup_size_list[i])):
            rgroup_mask_batch_new[cnt, tmp_idx[cnt][1]:tmp_idx[cnt][1]+rgroup_size_list[i][j], tmp_idx[cnt][2]] = 1
            cnt += 1
    out['rgroup_mask_batch_new'] = rgroup_mask_batch_new

    rgroup_mask_ori_batch_new = torch.repeat_interleave(rgroup_mask, batch_new_len_tensor, dim=0).to(rgroup_mask.device)#沿着行重复
    scaffold_mask_ori_batch_new = torch.repeat_interleave(scaffold_mask, batch_new_len_tensor, dim=0).to(rgroup_mask.device)
    affinity_label_batch_new = torch.repeat_interleave(affinity_label, batch_new_len_tensor, dim=0).to(rgroup_mask.device)   
    # edge_mask_batch_new = torch.repeat_interleave(edge_mask, batch_new_len_tensor, dim=0).to(rgroup_mask.device)
    node_mask_batch_new = torch.repeat_interleave(node_mask, batch_new_len_tensor, dim=0).to(rgroup_mask.device)
    scaffold_mask_batch_new = scaffold_mask_ori_batch_new + (rgroup_mask_ori_batch_new - rgroup_mask_batch_new)
    out['rgroup_mask_ori_batch_new'] = rgroup_mask_ori_batch_new
    #print(f"context shape: {out['rgroup_mask_ori_batch_new'].shape}")
    #exit()
    out['scaffold_mask_ori_batch_new'] = scaffold_mask_ori_batch_new
    out['node_mask_batch_new'] = node_mask_batch_new
    out['scaffold_mask_batch_new'] = scaffold_mask_batch_new

    out['affinity_label_batch_new'] = affinity_label_batch_new
    #print(f"scaffold_mask_batch_new shape: {out['affinity_label_batch_new'].shape}")
    #exit()

    
    pocket_mask_batch_new = torch.repeat_interleave(pocket_mask, batch_new_len_tensor, dim=0).to(rgroup_mask.device)
    scaffold_only_mask_batch_new = scaffold_mask_batch_new - pocket_mask_batch_new
    out['pocket_mask_batch_new'] = pocket_mask_batch_new
    out['scaffold_only_mask_batch_new'] = scaffold_only_mask_batch_new

    x_batch_new = torch.repeat_interleave(x, batch_new_len_tensor, dim=0).to(rgroup_mask.device)
    h_batch_new = torch.repeat_interleave(h, batch_new_len_tensor, dim=0).to(rgroup_mask.device)
    out['x_batch_new'] = x_batch_new
    out['h_batch_new'] = h_batch_new

    edge_mask = torch.cat([
        torch.ones(node_mask_batch_new.shape[1], dtype=torch.int8) * i
        for i in range(node_mask_batch_new.shape[0])
    ]).to(node_mask_batch_new.device)
    out['edge_mask'] = edge_mask

    return out