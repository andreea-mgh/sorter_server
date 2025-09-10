import os
import joblib
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# pentru reproducibilitate
RANDOM_STATE = 42


def retrain_svm(MODEL_NAME):
    DATA_DIR = MODEL_NAME

    image_paths, labels = [], []

    for label, class_name in enumerate(sorted(os.listdir(DATA_DIR))):
        class_dir = os.path.join(DATA_DIR, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(label)

    image_paths = np.array(image_paths)
    labels = np.array(labels)

    # ----------------------
    # Feature Extraction
    # ----------------------

    def extract_features(image_path):
        img = imread(image_path, as_gray=True)

        # HOG 
        hog_feat = hog(img, pixels_per_cell=(16,16), cells_per_block=(2,2),
                    orientations=9, block_norm='L2-Hys', feature_vector=True)

        img_uint8 = (img * 255).astype(np.uint8)
        # LBP 
        lbp = local_binary_pattern(img_uint8, P=8, R=1, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                bins=np.arange(0, 8 + 3),
                                range=(0, 8 + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)  # normalizare

        # HOG + LBP
        return np.hstack([hog_feat, hist])

    # print("Extracting features...")
    X = np.array([extract_features(p) for p in image_paths])
    y = labels

    # pca
    pca = PCA(n_components=100, random_state=RANDOM_STATE)
    X_reduced = pca.fit_transform(X)
    print("Reduced feature matrix shape:", X_reduced.shape)


    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # clasificator svm
    svm_clf = SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_STATE, probability=True)
    svm_clf.fit(X_train, y_train)
    y_pred_svm = svm_clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred_svm))
    print(classification_report(y_test, y_pred_svm))

    joblib.dump(svm_clf, f"models/{DATA_DIR}_svm_model.joblib")
    joblib.dump(pca, f"models/{DATA_DIR}_pca_transform.joblib")