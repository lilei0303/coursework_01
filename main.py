import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


# 定义函数封装整个机器学习流程
def fashion_mnist_classification():
    # 定义HOG特征提取函数
    def extract_hog_features(images):
        hog_features = []
        for image in images:
            # 计算 HOG 特征
            hog_feature = hog(image, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2),
                              transform_sqrt=True)
            hog_features.append(hog_feature)
        return np.array(hog_features)

    # 定义LBP特征提取函数
    def extract_lbp_features(images):
        lbp_features = []
        for image in images:
            # 提取LBP特征
            lbp_feature = local_binary_pattern(image, 8, 1, method='uniform').flatten()
            lbp_features.append(lbp_feature)
        return np.array(lbp_features)

    # 定义函数来训练和评估模型
    def train_and_evaluate_classifier(classifier, train_features, test_features):
        # 拟和数据
        classifier.fit(train_features, train_labels)
        # 在测试集上进行预测
        predicted_labels = classifier.predict(test_features)
        # 计算准确率
        accuracy = accuracy_score(test_labels, predicted_labels)
        return accuracy

    # 定义数据集的下载和转换
    def get_train_datas():
        # 将图像转换为张量
        transform = transforms.ToTensor()
        # 加载FashionMNIST数据集
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        # 获取图像和标签
        train_images = train_dataset.data.numpy()
        train_labels = train_dataset.targets.numpy()
        test_images = test_dataset.data.numpy()
        test_labels = test_dataset.targets.numpy()
        return train_images, train_labels, test_images, test_labels

    # 获取数据
    train_images, train_labels, test_images, test_labels = get_train_datas()

    print("正在提取HOG特征...")
    # 提取HOG特征
    train_hog_features = extract_hog_features(train_images)
    test_hog_features = extract_hog_features(test_images)

    print("正在提取LBP特征...")
    # 提取LBP特征
    train_lbp_features = extract_lbp_features(train_images)
    test_lbp_features = extract_lbp_features(test_images)

    print("特征提取完成，准备训练...")

    # 存储各个训练模型的准确率
    accuracies = {}

    # 使用SVM分类器，多项式核函数，3次多项式
    print("\nSVM with Polynomial Kernel starting...:")
    svm_poly = svm.SVC(kernel='poly', degree=3)
    accuracies['svm_poly_hog'] = train_and_evaluate_classifier(svm_poly, train_hog_features, test_hog_features)
    accuracies['svm_poly_lbp'] = train_and_evaluate_classifier(svm_poly, train_lbp_features, test_lbp_features)

    # 使用SVM分类器，高斯核函数
    print("\nSVM with RBF Kernel starting...:")
    svm_rbf = svm.SVC(kernel='rbf')
    accuracies['svm_rbf_hog'] = train_and_evaluate_classifier(svm_rbf, train_hog_features, test_hog_features)
    accuracies['svm_rbf_lbp'] = train_and_evaluate_classifier(svm_rbf, train_lbp_features, test_lbp_features)

    # 使用前馈神经网络分类器
    print("\nFeedforward Neural Network starting...:")
    mlp = MLPClassifier(hidden_layer_sizes=(200, 100, 50), max_iter=1000, alpha=1e-3,
                        solver='adam', verbose=10, tol=1e-4, random_state=1,
                        learning_rate_init=0.001)
    accuracies['mlp_hog'] = train_and_evaluate_classifier(mlp, train_hog_features, test_hog_features)
    accuracies['mlp_lbp'] = train_and_evaluate_classifier(mlp, train_lbp_features, test_lbp_features)

    # 打印所有训练模型的预测精确度
    for key, value in accuracies.items():
        print("{}: {}".format(key, value))


# 调用封装好的函数
fashion_mnist_classification()
