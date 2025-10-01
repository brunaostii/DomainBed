#!/usr/bin/env python3
"""
Script abrangente para validar se o dataset ISIC está sendo carregado corretamente
no DomainBed e se o algoritmo está sendo aplicado adequadamente.
"""

import sys
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import seaborn as sns

def check_data_structure():
    """Verifica a estrutura dos dados e paths"""
    print("=== VERIFICAÇÃO DA ESTRUTURA DOS DADOS ===")
    
    # Paths esperados
    root_path = "../../"
    data_path = os.path.join(root_path, "data/ham10000/HAM10000/")
    splits_path = os.path.join(root_path, "work/causality_master/datasplits/ham10000/splits/")
    
    print(f"Root path: {root_path}")
    print(f"Data path: {data_path}")
    print(f"Splits path: {splits_path}")
    
    # Verificar se os diretórios existem
    print(f"\n📁 Verificação de diretórios:")
    print(f"  Data path exists: {os.path.exists(data_path)}")
    print(f"  Splits path exists: {os.path.exists(splits_path)}")
    
    if os.path.exists(data_path):
        # Contar quantas imagens há no diretório
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = []
        for file in os.listdir(data_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file)
        print(f"  Number of image files found: {len(image_files)}")
        if len(image_files) > 0:
            print(f"  Sample files: {image_files[:10]}")
    
    if os.path.exists(splits_path):
        # Verificar arquivos de split
        split_files = [f for f in os.listdir(splits_path) if f.endswith('.csv')]
        print(f"  Number of split files: {len(split_files)}")
        print(f"  Split files: {split_files}")
    
    return data_path, splits_path

def analyze_split_files(splits_path):
    """Analisa os arquivos de split para entender a distribuição dos dados"""
    print("\n=== ANÁLISE DOS ARQUIVOS DE SPLIT ===")
    
    environments = ["baseline", "size_20_20", "size_80_20", "size_50_50"]
    
    for env in environments:
        print(f"\n📊 Environment: {env}")
        
        train_file = os.path.join(splits_path, f"train_{env}.csv")
        val_file = os.path.join(splits_path, f"val_{env}.csv")
        
        if os.path.exists(train_file) and os.path.exists(val_file):
            train_df = pd.read_csv(train_file)
            val_df = pd.read_csv(val_file)
            
            print(f"  Train samples: {len(train_df)}")
            print(f"  Val samples: {len(val_df)}")
            print(f"  Total samples: {len(train_df) + len(val_df)}")
            
            # Verificar colunas
            print(f"  Train columns: {list(train_df.columns)}")
            
            # Verificar distribuição de classes
            if 'label' in train_df.columns:
                train_dist = train_df['label'].value_counts()
                val_dist = val_df['label'].value_counts()
                print(f"  Train label distribution: {dict(train_dist)}")
                print(f"  Val label distribution: {dict(val_dist)}")
            
            # Verificar algumas amostras
            print(f"  Sample train entries:")
            print(train_df.head(3))
        else:
            print(f"  ❌ Files not found: {train_file}, {val_file}")

def test_image_loading(data_path, splits_path):
    """Testa o carregamento de imagens"""
    print("\n=== TESTE DE CARREGAMENTO DE IMAGENS ===")
    
    # Carregar um arquivo de split para teste
    train_file = os.path.join(splits_path, "train_baseline.csv")
    
    if not os.path.exists(train_file):
        print(f"❌ Arquivo de treino não encontrado: {train_file}")
        return False
    
    df = pd.read_csv(train_file)
    print(f"✅ Split file loaded: {len(df)} samples")
    
    # Testar carregamento de algumas imagens
    success_count = 0
    fail_count = 0
    tested_images = []
    
    for idx in range(min(10, len(df))):  # Testar as primeiras 10 imagens
        row = df.iloc[idx]
        image_id = row['image_id']
        label = row['label']
        
        # Tentar diferentes extensões
        image_path = None
        for ext in ['.jpg', '_downsampled.jpg', '.jpeg', '.png']:
            potential_path = os.path.join(data_path, f"{image_id}{ext}")
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if image_path:
            try:
                image = Image.open(image_path).convert('RGB')
                tested_images.append({
                    'image_id': image_id,
                    'label': label,
                    'path': image_path,
                    'size': image.size
                })
                success_count += 1
                print(f"  ✅ {image_id}: {image.size} - label: {label}")
            except Exception as e:
                print(f"  ❌ Error loading {image_id}: {e}")
                fail_count += 1
        else:
            print(f"  ❌ Image not found: {image_id}")
            fail_count += 1
    
    print(f"\n📊 Image loading results:")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Success rate: {success_count/(success_count+fail_count)*100:.1f}%")
    
    return tested_images

def test_transforms(tested_images):
    """Testa as transformações aplicadas às imagens"""
    print("\n=== TESTE DE TRANSFORMAÇÕES ===")
    
    if not tested_images:
        print("❌ Nenhuma imagem disponível para teste")
        return
    
    # Definir transformações (iguais às do dataset)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    augment_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Testar transformações
    sample_image = tested_images[0]
    image = Image.open(sample_image['path']).convert('RGB')
    
    print(f"Original image size: {image.size}")
    
    # Aplicar transformação básica
    transformed = transform(image)
    print(f"✅ Basic transform: {transformed.shape}, dtype: {transformed.dtype}")
    print(f"  Min value: {transformed.min():.3f}, Max value: {transformed.max():.3f}")
    
    # Aplicar transformação com augmentação
    augmented = augment_transform(image)
    print(f"✅ Augmented transform: {augmented.shape}, dtype: {augmented.dtype}")
    print(f"  Min value: {augmented.min():.3f}, Max value: {augmented.max():.3f}")

def test_dataset_class():
    """Testa a classe do dataset"""
    print("\n=== TESTE DA CLASSE DATASET ===")
    
    try:
        # Importar e testar a classe
        from domainbed.datasets import ISIC2019DomainBed
        
        root_path = "../../"
        test_envs = [0]  # Usar primeiro ambiente como teste
        hparams = {'data_augmentation': True}
        
        print("Inicializando dataset...")
        dataset = ISIC2019DomainBed(root_path, test_envs, hparams)
        
        print(f"✅ Dataset inicializado com sucesso!")
        print(f"  Número de ambientes: {len(dataset)}")
        print(f"  Nomes dos ambientes: {dataset.ENVIRONMENTS}")
        print(f"  Formato de entrada: {dataset.input_shape}")
        print(f"  Número de classes: {dataset.num_classes}")
        
        # Testar cada ambiente
        for i, env_dataset in enumerate(dataset.datasets):
            env_name = dataset.ENVIRONMENTS[i]
            print(f"\n🌍 Ambiente {i} ({env_name}):")
            print(f"  Número de amostras: {len(env_dataset)}")
            
            if len(env_dataset) > 0:
                # Testar carregamento de amostra
                try:
                    sample, label = env_dataset[0]
                    print(f"  ✅ Amostra carregada com sucesso")
                    print(f"    Shape: {sample.shape}")
                    print(f"    Label: {label}")
                    print(f"    Dtype: {sample.dtype}")
                    print(f"    Min: {sample.min():.3f}, Max: {sample.max():.3f}")
                    
                    # Testar algumas amostras aleatórias
                    for j in range(min(5, len(env_dataset))):
                        sample, label = env_dataset[j]
                        if j == 0:
                            print(f"    Primeiras 5 labels: ", end="")
                        print(f"{label}", end=" ")
                    print()
                    
                except Exception as e:
                    print(f"  ❌ Erro ao carregar amostra: {e}")
                    import traceback
                    traceback.print_exc()
        
        return dataset
        
    except Exception as e:
        print(f"❌ Erro ao testar dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_dataloader_integration(dataset):
    """Testa a integração com DataLoader"""
    print("\n=== TESTE DE INTEGRAÇÃO COM DATALOADER ===")
    
    if dataset is None:
        print("❌ Dataset não disponível")
        return
    
    from torch.utils.data import DataLoader
    
    # Testar DataLoader para cada ambiente
    for i, env_dataset in enumerate(dataset.datasets):
        env_name = dataset.ENVIRONMENTS[i]
        print(f"\n🔄 Testando DataLoader para ambiente {env_name}:")
        
        try:
            loader = DataLoader(
                dataset=env_dataset,
                batch_size=4,
                shuffle=True,
                num_workers=0  # Evitar problemas de multiprocessing
            )
            
            # Testar um batch
            batch_x, batch_y = next(iter(loader))
            print(f"  ✅ Batch carregado com sucesso")
            print(f"    Batch X shape: {batch_x.shape}")
            print(f"    Batch Y shape: {batch_y.shape}")
            print(f"    Batch Y values: {batch_y.tolist()}")
            print(f"    Batch X dtype: {batch_x.dtype}")
            print(f"    Batch Y dtype: {batch_y.dtype}")
            
        except Exception as e:
            print(f"  ❌ Erro no DataLoader: {e}")
            import traceback
            traceback.print_exc()

def test_algorithm_integration(dataset):
    """Testa a integração com algoritmos do DomainBed"""
    print("\n=== TESTE DE INTEGRAÇÃO COM ALGORITMOS ===")
    
    if dataset is None:
        print("❌ Dataset não disponível")
        return
    
    try:
        from domainbed import algorithms, hparams_registry
        
        # Testar com algoritmo ERM
        algorithm_name = "ERM"
        print(f"Testando algoritmo: {algorithm_name}")
        
        # Obter hiperparâmetros
        hparams = hparams_registry.default_hparams(algorithm_name, "ISIC2019DomainBed")
        hparams['batch_size'] = 4  # Usar batch pequeno para teste
        print(f"Hiperparâmetros: {hparams}")
        
        # Criar algoritmo
        algorithm_class = algorithms.get_algorithm_class(algorithm_name)
        test_envs = [0]
        num_train_envs = len(dataset) - len(test_envs)
        
        algorithm = algorithm_class(
            dataset.input_shape, 
            dataset.num_classes, 
            num_train_envs, 
            hparams
        )
        
        # Mover para dispositivo apropriado
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        algorithm = algorithm.to(device)
        print(f"✅ Algoritmo criado e movido para: {device}")
        
        # Testar um passo de treinamento
        from torch.utils.data import DataLoader
        
        train_envs = [i for i in range(len(dataset)) if i not in test_envs]
        
        # Criar mini-batches
        minibatches = []
        for env_idx in train_envs:
            loader = DataLoader(
                dataset=dataset[env_idx],
                batch_size=hparams['batch_size'],
                shuffle=True,
                num_workers=0
            )
            batch_x, batch_y = next(iter(loader))
            minibatches.append([batch_x.to(device), batch_y.to(device)])
        
        print(f"Criados {len(minibatches)} mini-batches para treinamento")
        
        # Executar um passo de atualização
        algorithm.train()
        loss = algorithm.update(minibatches)
        
        if isinstance(loss, dict):
            print(f"✅ Passo de treinamento bem-sucedido! Loss: {loss}")
        else:
            print(f"✅ Passo de treinamento bem-sucedido! Loss: {loss:.4f}")
        
        # Testar predição
        algorithm.eval()
        test_env_idx = test_envs[0]
        test_loader = DataLoader(
            dataset=dataset[test_env_idx],
            batch_size=hparams['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        test_x, test_y = next(iter(test_loader))
        test_x, test_y = test_x.to(device), test_y.to(device)
        
        with torch.no_grad():
            predictions = algorithm.predict(test_x)
            print(f"✅ Predições bem-sucedidas! Shape: {predictions.shape}")
            print(f"    Primeiras predições: {predictions[:3].argmax(dim=1).tolist()}")
            print(f"    Labels reais: {test_y[:3].tolist()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao testar algoritmo: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Função principal que executa todos os testes"""
    print("🔍 VALIDAÇÃO COMPLETA DO DATASET ISIC NO DOMAINBED")
    print("=" * 60)
    
    # 1. Verificar estrutura dos dados
    data_path, splits_path = check_data_structure()
    
    # 2. Analisar arquivos de split
    analyze_split_files(splits_path)
    
    # 3. Testar carregamento de imagens
    tested_images = test_image_loading(data_path, splits_path)
    
    # 4. Testar transformações
    test_transforms(tested_images)
    
    # 5. Testar classe do dataset
    dataset = test_dataset_class()
    
    # 6. Testar integração com DataLoader
    test_dataloader_integration(dataset)
    
    # 7. Testar integração com algoritmos
    algorithm_success = test_algorithm_integration(dataset)
    
    # Resumo final
    print("\n" + "=" * 60)
    print("📋 RESUMO DA VALIDAÇÃO")
    print("=" * 60)
    
    if tested_images:
        print("✅ Carregamento de imagens: OK")
    else:
        print("❌ Carregamento de imagens: FALHOU")
    
    if dataset is not None:
        print("✅ Classe do dataset: OK")
    else:
        print("❌ Classe do dataset: FALHOU")
    
    if algorithm_success:
        print("✅ Integração com algoritmos: OK")
    else:
        print("❌ Integração com algoritmos: FALHOU")
    
    print("\n🎯 RECOMENDAÇÕES:")
    print("1. Verifique se as imagens estão no formato correto")
    print("2. Confirme se os paths nos arquivos CSV estão corretos")
    print("3. Teste com diferentes algoritmos do DomainBed")
    print("4. Monitore o desempenho durante o treinamento")

if __name__ == "__main__":
    main()
