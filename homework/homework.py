# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import pandas as pd
import os
import gzip
import pickle
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Any
from dataclasses import dataclass

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuración centralizada del proyecto"""

    INPUT_PATH: str = "files/input/"
    MODELS_PATH: str = "files/models/"
    OUTPUT_PATH: str = "files/output/"
    TRAIN_FILE: str = "train_data.csv.zip"
    TEST_FILE: str = "test_data.csv.zip"
    MODEL_FILE: str = "model.pkl.gz"
    METRICS_FILE: str = "metrics.json"
    RANDOM_STATE: int = 42
    CV_FOLDS: int = 10
    CATEGORICAL_FEATURES: list = None

    def __post_init__(self):
        if self.CATEGORICAL_FEATURES is None:
            self.CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]


class DataLoader:
    """Clase para cargar y procesar datos"""

    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """
        Carga datos desde un archivo CSV comprimido

        Args:
            file_path: Ruta al archivo CSV comprimido

        Returns:
            DataFrame con los datos cargados

        Raises:
            FileNotFoundError: Si el archivo no existe
            Exception: Para otros errores de carga
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"El archivo {file_path} no existe")

            logger.info(f"Cargando datos desde {file_path}")
            return pd.read_csv(file_path, index_col=False, compression="zip")

        except Exception as e:
            logger.error(f"Error al cargar datos desde {file_path}: {str(e)}")
            raise

    @staticmethod
    def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia y preprocesa el dataset

        Args:
            df: DataFrame a limpiar

        Returns:
            DataFrame limpio
        """
        logger.info("Iniciando limpieza del dataset")

        # Crear copia para evitar modificar el original
        df_clean = df.copy()

        # Renombrar columna target
        df_clean = df_clean.rename(columns={"default payment next month": "default"})

        # Eliminar columna ID
        df_clean = df_clean.drop(columns=["ID"])

        # Filtrar valores inválidos
        df_clean = df_clean.loc[df_clean["MARRIAGE"] != 0]
        df_clean = df_clean.loc[df_clean["EDUCATION"] != 0]

        # Agrupar educación: valores >= 4 se convierten en 4
        df_clean["EDUCATION"] = df_clean["EDUCATION"].apply(lambda x: x if x < 4 else 4)

        logger.info(f"Dataset limpio. Shape: {df_clean.shape}")
        return df_clean

    @staticmethod
    def split_features_target(
        df: pd.DataFrame, target_col: str = "default"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separa características y variable objetivo

        Args:
            df: DataFrame completo
            target_col: Nombre de la columna objetivo

        Returns:
            Tupla con (características, objetivo)
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return X, y


class ModelBuilder:
    """Clase para construir y entrenar modelos"""

    def __init__(self, config: Config):
        self.config = config

    def create_pipeline(self) -> Pipeline:
        """
        Crea pipeline de preprocessing y clasificación

        Returns:
            Pipeline configurado
        """
        logger.info("Creando pipeline de procesamiento")

        # Preprocessor para variables categóricas
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    self.config.CATEGORICAL_FEATURES,
                )
            ],
            remainder="passthrough",
        )

        # Pipeline completo
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(random_state=self.config.RANDOM_STATE),
                ),
            ]
        )

        return pipeline

    def create_grid_search(self, pipeline: Pipeline) -> GridSearchCV:
        """
        Crea GridSearchCV para optimización de hiperparámetros

        Args:
            pipeline: Pipeline base

        Returns:
            GridSearchCV configurado
        """
        logger.info("Configurando GridSearchCV")

        param_grid = {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [None, 5, 10, 20],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
        }

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=self.config.CV_FOLDS,
            scoring="balanced_accuracy",
            n_jobs=-1,
            verbose=2,
            refit=True,
        )

        return grid_search

    def save_model(self, model: GridSearchCV, file_path: str) -> None:
        """
        Guarda modelo entrenado

        Args:
            model: Modelo entrenado
            file_path: Ruta donde guardar el modelo
        """
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            logger.info(f"Guardando modelo en {file_path}")
            with gzip.open(file_path, "wb") as f:
                pickle.dump(model, f)

            logger.info("Modelo guardado exitosamente")

        except Exception as e:
            logger.error(f"Error al guardar modelo: {str(e)}")
            raise


class MetricsCalculator:
    """Clase para calcular métricas de evaluación"""

    @staticmethod
    def calculate_classification_metrics(
        dataset_name: str, y_true: pd.Series, y_pred: pd.Series
    ) -> Dict[str, Any]:
        """
        Calcula métricas de clasificación

        Args:
            dataset_name: Nombre del dataset
            y_true: Valores reales
            y_pred: Valores predichos

        Returns:
            Diccionario con métricas
        """
        return {
            "type": "metrics",
            "dataset": dataset_name,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }

    @staticmethod
    def calculate_confusion_matrix_metrics(
        dataset_name: str, y_true: pd.Series, y_pred: pd.Series
    ) -> Dict[str, Any]:
        """
        Calcula matriz de confusión

        Args:
            dataset_name: Nombre del dataset
            y_true: Valores reales
            y_pred: Valores predichos

        Returns:
            Diccionario con matriz de confusión
        """
        cm = confusion_matrix(y_true, y_pred)
        return {
            "type": "cm_matrix",
            "dataset": dataset_name,
            "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
            "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])},
        }

    def save_metrics(self, metrics_list: list, file_path: str) -> None:
        """
        Guarda métricas en archivo JSON

        Args:
            metrics_list: Lista de métricas
            file_path: Ruta del archivo
        """
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            logger.info(f"Guardando métricas en {file_path}")
            with open(file_path, "w") as file:
                for metric in metrics_list:
                    file.write(json.dumps(metric) + "\n")

            logger.info("Métricas guardadas exitosamente")

        except Exception as e:
            logger.error(f"Error al guardar métricas: {str(e)}")
            raise


class CreditClassifier:
    """Clase principal que orquesta todo el proceso"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.data_loader = DataLoader()
        self.model_builder = ModelBuilder(self.config)
        self.metrics_calculator = MetricsCalculator()

    def run(self) -> None:
        """Ejecuta todo el pipeline de entrenamiento y evaluación"""
        try:
            logger.info("Iniciando proceso de clasificación de crédito")

            # 1. Cargar datos
            train_df = self.data_loader.load_data(
                os.path.join(self.config.INPUT_PATH, self.config.TRAIN_FILE)
            )
            test_df = self.data_loader.load_data(
                os.path.join(self.config.INPUT_PATH, self.config.TEST_FILE)
            )

            # 2. Limpiar datos
            train_df_clean = self.data_loader.clean_dataset(train_df)
            test_df_clean = self.data_loader.clean_dataset(test_df)

            # 3. Separar características y objetivo
            X_train, y_train = self.data_loader.split_features_target(train_df_clean)
            X_test, y_test = self.data_loader.split_features_target(test_df_clean)

            # 4. Crear y entrenar modelo
            pipeline = self.model_builder.create_pipeline()
            grid_search = self.model_builder.create_grid_search(pipeline)

            logger.info("Entrenando modelo...")
            grid_search.fit(X_train, y_train)
            logger.info("Entrenamiento completado")

            # 5. Guardar modelo
            model_path = os.path.join(self.config.MODELS_PATH, self.config.MODEL_FILE)
            self.model_builder.save_model(grid_search, model_path)

            # 6. Realizar predicciones
            logger.info("Realizando predicciones...")
            y_train_pred = grid_search.predict(X_train)
            y_test_pred = grid_search.predict(X_test)

            # 7. Calcular métricas
            train_metrics = self.metrics_calculator.calculate_classification_metrics(
                "train", y_train, y_train_pred
            )
            test_metrics = self.metrics_calculator.calculate_classification_metrics(
                "test", y_test, y_test_pred
            )
            train_cm = self.metrics_calculator.calculate_confusion_matrix_metrics(
                "train", y_train, y_train_pred
            )
            test_cm = self.metrics_calculator.calculate_confusion_matrix_metrics(
                "test", y_test, y_test_pred
            )

            # 8. Guardar métricas
            all_metrics = [train_metrics, test_metrics, train_cm, test_cm]
            metrics_path = os.path.join(
                self.config.OUTPUT_PATH, self.config.METRICS_FILE
            )
            self.metrics_calculator.save_metrics(all_metrics, metrics_path)

            # 9. Mostrar resultados
            logger.info("Proceso completado exitosamente")
            logger.info(f"Mejor score: {grid_search.best_score_:.4f}")
            logger.info(f"Mejores parámetros: {grid_search.best_params_}")

        except Exception as e:
            logger.error(f"Error en el proceso principal: {str(e)}")
            raise


def main():
    """Función principal"""
    try:
        # Crear instancia del clasificador con configuración por defecto
        classifier = CreditClassifier()

        # Ejecutar el proceso completo
        classifier.run()

    except Exception as e:
        logger.error(f"Error fatal: {str(e)}")
        raise


if __name__ == "__main__":
    main()
