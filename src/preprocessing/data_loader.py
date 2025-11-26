from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import pandas as pd
import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class TCGADataLoader:
    def __init__(self, config_path: str = None, use_cache: bool = True):
        """Initialize the data loader with configuration."""
        self.base_dir = Path(__file__).resolve().parent.parent.parent

        config_path = Path(config_path or "config/config.yaml")
        if not config_path.is_absolute():
            config_path = self.base_dir / config_path

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        data_cfg = self.config['data']
        self.expression_path = Path(data_cfg['expression_path']).expanduser()
        self.mutation_path = Path(data_cfg['mutation_path']).expanduser()
        self.gene_name_mapping_path = Path(data_cfg['gene_name_mapping_path']).expanduser()
        self.gene_annotation_path = Path(data_cfg['gene_annotation_path']).expanduser()

        for path_attr in ("expression_path", "mutation_path"):
            path = getattr(self, path_attr)
            path.mkdir(parents=True, exist_ok=True)

        for path_attr in ("gene_name_mapping_path", "gene_annotation_path"):
            resolved = getattr(self, path_attr)
            if not resolved.exists():
                raise FileNotFoundError(f"Configured path does not exist: {resolved}")

        self.use_cache = use_cache
        self.cache_dir = self.base_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)

        cancer_types_cfg = data_cfg.get('cancer_types') or []
        self.default_cancer_types = [ct.upper() for ct in cancer_types_cfg] or None

        self.download_missing = data_cfg.get('download_missing', True)
        self.expression_base_url = data_cfg.get('expression_base_url')
        self.mutation_base_url = data_cfg.get('mutation_base_url')
        self.expression_file_template = data_cfg.get('expression_file_template', 'TCGA-{cancer}.star_{measure}.tsv.gz')
        self.mutation_file_template = data_cfg.get('mutation_file_template', 'mc3_gene_level%2F{cancer}_mc3_gene_level.txt.gz')

        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

    # def _download_gencode(self) -> Path:
    #     """Download GENCODE annotation file if not already present."""
    #     gencode_path = self.cache_dir / "gencode.v44.annotation.gff3.gz"
    #     if not gencode_path.exists():
    #         print("Downloading GENCODE annotation file...")
    #         url = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gff3.gz"
    #         response = requests.get(url, stream=True)
    #         total_size = int(response.headers.get('content-length', 0))
            
    #         with open(gencode_path, 'wb') as f, tqdm(
    #             desc="Downloading",
    #             total=total_size,
    #             unit='iB',
    #             unit_scale=True
    #         ) as pbar:
    #             for data in response.iter_content(chunk_size=1024):
    #                 size = f.write(data)
    #                 pbar.update(size)
        
    #     return gencode_path
    
    # def _get_protein_coding_genes(self) -> Set[str]:
    #     """Get set of protein-coding gene symbols from GENCODE."""
    #     gencode_path = self._download_gencode()
    #     db_path = self.cache_dir / "gencode.db"
        
    #     # Create database if it doesn't exist
    #     if not db_path.exists():
    #         print("Creating GENCODE database...")
    #         db = gffutils.create_db(
    #             str(gencode_path),
    #             str(db_path),
    #             force=True,
    #             keep_order=True,
    #             merge_strategy='merge',
    #             sort_attribute_values=True
    #         )
    #     else:
    #         db = gffutils.FeatureDB(str(db_path))
        
    #     # Get protein-coding genes
    #     protein_coding_genes = set()
    #     for gene in db.features_of_type('gene'):
    #         if gene.attributes.get('gene_type', [''])[0] == 'protein_coding':
    #             gene_name = gene.attributes.get('gene_name', [''])[0]
    #             if gene_name:
    #                 protein_coding_genes.add(gene_name)
        
    #     return protein_coding_genes

    def load_expression_data(self, load_tpm: bool = False, cancer_types: Optional[List[str]] = None) -> pd.DataFrame:
        measure = 'tpm' if (load_tpm or self.config.get('data', {}).get('use_tpm', False)) else 'counts'
        gene_name_mapping_df = pd.read_csv(self.gene_name_mapping_path, sep='\t', index_col=0)

        expression_files = self._resolve_expression_files(measure=measure, cancer_types=cancer_types)

        counts_df_list = []
        for file_path in expression_files:
            counts_df = pd.read_csv(file_path, sep='\t', index_col=0).T
            if self.config['data'].get('log_transformed', True):
                counts_df = (2**counts_df - 1)
            cancer_type = self._infer_cancer_from_expression(file_path.name)
            counts_df['cancer_type'] = cancer_type
            counts_df_list.append(counts_df)

        counts_df = pd.concat(counts_df_list)
        cancer_type_series = counts_df['cancer_type']
        counts_df.drop('cancer_type', axis=1, inplace=True)
        counts_df.columns = counts_df.columns + '|' + counts_df.columns.map(gene_name_mapping_df['gene'])
        if self.config['data'].get('only_protein_coding_expression', True):
            gene_annotation_df = pd.read_csv(
                self.gene_annotation_path,
                sep='\t',
                compression='gzip',
                names=['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute'],
                comment='#'
            )
            gene_annotation_df['gene_id'] = gene_annotation_df['attribute'].str.extract('gene_id "([^"]*)"')
            gene_annotation_df['gene_type'] = gene_annotation_df['attribute'].str.extract('gene_type "([^"]*)"')
            annotation_map = dict(zip(gene_annotation_df['gene_id'], gene_annotation_df['gene_type']))
            gene_types = counts_df.columns.str.split('|').str[0].map(annotation_map)
            counts_df = counts_df.loc[:, gene_types == 'protein_coding']

        counts_df['cancer_type'] = cancer_type_series
        cancer_type_dummies = pd.get_dummies(counts_df['cancer_type'], prefix='cancer', dtype=int)
        counts_df.drop('cancer_type', axis=1, inplace=True)
        counts_df = pd.concat([counts_df, cancer_type_dummies], axis=1)

        return counts_df

    def load_mutation_data(self, cancer_types: Optional[List[str]] = None) -> pd.DataFrame:
        mutation_files = self._resolve_mutation_files(cancer_types=cancer_types)
        mutation_df_list = []
        for file_path in mutation_files:
            mutation_df = pd.read_csv(file_path, sep='\t', compression='gzip', index_col=0).T
            mutation_df_list.append(mutation_df)
        mutation_df = pd.concat(mutation_df_list)
        # cancer_type_series = mutation_df['cancer_type']
        if self.config['data'].get('only_protein_coding_mutations', True):
            gene_annotation_df = pd.read_csv(self.gene_annotation_path, sep='\t', compression='gzip', names=['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute'], comment='#')
            gene_annotation_df['gene_type'] = gene_annotation_df['attribute'].str.extract('gene_type "([^"]*)"')
            gene_annotation_df['gene_name'] = gene_annotation_df['attribute'].str.extract('gene_name "([^"]*)"')
            annotation_map = dict(zip(gene_annotation_df['gene_name'], gene_annotation_df['gene_type']))
            gene_types = mutation_df.columns.map(annotation_map)
            mutation_df = mutation_df.loc[:, gene_types == 'protein_coding']
        min_mutations = self.config['data'].get('min_mutations_per_gene', 1)
        mutation_counts = mutation_df.sum()
        if isinstance(min_mutations, float) and 0 < min_mutations < 1:
            min_sample_count = len(mutation_df) * min_mutations
            mutation_df = mutation_df.loc[:, mutation_counts >= min_sample_count]
        else:
            mutation_df = mutation_df.loc[:, mutation_counts >= min_mutations]
        # mutation_df['cancer_type'] = cancer_type_series
        return mutation_df

    def align_data(self, expression_data: pd.DataFrame, mutation_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        mutation_data.index = mutation_data.index + 'A'
        common_samples = expression_data.index.intersection(mutation_data.index)
        return expression_data.loc[common_samples], mutation_data.loc[common_samples]

    def preprocess_data(self, load_tpm: bool = False, cancer_types: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cohorts = self._normalize_cancer_types(cancer_types)
        expr_cache_file, mut_cache_file = self._cache_paths(cohorts)

        if self.use_cache and expr_cache_file.exists() and mut_cache_file.exists():
            print("Loading cached aligned expression and mutation data from:")
            print(expr_cache_file, mut_cache_file)
            expression_data = joblib.load(expr_cache_file)
            mutation_data = joblib.load(mut_cache_file)
        else:
            print("Processing data from raw files...")
            print("Loading expression from: " + str(self.expression_path))
            print("Loading mutations from: " + str(self.mutation_path))
            expression_data = self.load_expression_data(load_tpm, cancer_types=cohorts)
            mutation_data = self.load_mutation_data(cancer_types=cohorts)
            expression_data, mutation_data = self.align_data(expression_data, mutation_data)
            expression_data = expression_data.fillna(0)
            mutation_data = mutation_data.fillna(0)
            if self.use_cache:
                joblib.dump(expression_data, expr_cache_file)
                joblib.dump(mutation_data, mut_cache_file)
        return expression_data, mutation_data

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalize_cancer_types(self, cancer_types: Optional[List[str]]) -> Optional[List[str]]:
        if cancer_types:
            return [ct.upper() for ct in cancer_types]
        return self.default_cancer_types

    def _cohort_cache_key(self, cancer_types: Optional[List[str]]) -> str:
        if not cancer_types:
            return "all"
        return "-".join(sorted(cancer_types))

    def _cache_paths(self, cancer_types: Optional[List[str]]) -> Tuple[Path, Path]:
        key = self._cohort_cache_key(cancer_types)
        expression_cache_file = self.cache_dir / f"expression_aligned_{key}.pkl"
        mutation_cache_file = self.cache_dir / f"mutation_aligned_{key}.pkl"
        return expression_cache_file, mutation_cache_file

    def _resolve_expression_files(self, measure: str, cancer_types: Optional[List[str]]) -> List[Path]:
        if cancer_types:
            files = []
            for cancer in cancer_types:
                filename = self.expression_file_template.format(cancer=cancer, measure=measure)
                file_path = self.expression_path / filename
                self._ensure_file(file_path, base_url=self.expression_base_url)
                files.append(file_path)
            return files

        suffix = f"star_{measure}.tsv.gz"
        return [
            self.expression_path / f
            for f in os.listdir(self.expression_path)
            if f.endswith(suffix)
        ]

    def _resolve_mutation_files(self, cancer_types: Optional[List[str]]) -> List[Path]:
        if cancer_types:
            files = []
            for cancer in cancer_types:
                filename = self.mutation_file_template.format(cancer=cancer)
                file_path = self.mutation_path / filename
                self._ensure_file(file_path, base_url=self.mutation_base_url)
                files.append(file_path)
            return files

        return [
            self.mutation_path / f
            for f in os.listdir(self.mutation_path)
            if f.endswith('gene_level.txt.gz')
        ]

    @staticmethod
    def _infer_cancer_from_expression(filename: str) -> str:
        try:
            return filename.split('-')[1].split('.')[0].upper()
        except IndexError:
            return filename

    @staticmethod
    def _session() -> requests.Session:
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
        session.mount("https://", HTTPAdapter(max_retries=retries))
        return session

    def _download_file(self, url: str, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with self._session().get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(out_path, 'wb') as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        handle.write(chunk)

    def _ensure_file(self, path: Path, base_url: Optional[str]) -> None:
        if path.exists():
            return
        if not self.download_missing or not base_url:
            raise FileNotFoundError(f"Required file not found: {path}")
        url = f"{base_url.rstrip('/')}/{path.name}"
        self.logger.info(f"Downloading {path.name} from {url}")
        self._download_file(url, path)
