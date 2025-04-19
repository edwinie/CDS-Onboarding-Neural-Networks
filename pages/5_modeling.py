import streamlit as st

st.title("Modeling")

st.header("Regression Question")
st.write("Can we predict a song's popularity using features such as danceability, energy, key, etc?")

st.header("K-Nearest Neighbor Regressor")
st.subheader("How does a k-NN work? ")
st.write("k-NN finds the k closest songs based on feature similarity and averages their popularity to predict for new songs.")

st.subheader("What’s the tradeoff between making k smaller or larger?")
st.write("""
Small k makes the model very sensitive to noise and overfits the data. Large k leads to underfitting.
The best k minimizes validation MSE. When k = 1, training MSE is low but validation MSE is high.
The lowest validation MSE occurs around k = 10, which means it generalizes well to unseen data.
""")

st.image("streamlitImages/knnq1.png")

st.subheader("What happens when you reduce the number of input features?")
st.write("""
         If MSE increases then removed features were important. If MSE stays the same or 
decreases then removed features were irrelevant or noisy. The validation MSE increased
when switching from All Features to Reduced Features. This suggests that the removed
features contained useful information that helped improve predictions.""")

st.image("streamlitImages/knnq2.png")
st.image("streamlitImages/knnfeatureremoval.png")

st.subheader("""What happens when you normalize your input data? If it is already normalized, what happens when you scale your input data to different proportions?""")

st.write("""
Normalization can impact k-NN regression performance but its effectiveness depends
on the dataset. In our graph, all three SE values are quite similar which suggests that 
the feature magnitudes were already balanced so normalization didn't make a large 
difference. If normalization had reduced MSE significantly, it would mean that scaling 
improved feature comparability and helped k-NN make better predictions. Since MSE didn't 
significantly decrease, this suggests that scaling wasn’t critical for this dataset's 
k-NN performance.
         """)

st.image("streamlitImages/knnq3.png")

st.header("Support Vector Regressor")

st.subheader("How does an SVR work?")
st.write("Support Vector Regression predicts continuous outcomes by fitting a hyperplane within an acceptable margin, penalizing errors only beyond that boundary. It uses kernel functions to transform data into higher dimensions, enabling it to capture nonlinear relationships.")

st.subheader("How is this similar to or different from linear regression? What do the different kernel types between linear, polynomial, and radial basis function (RBF) do?")
st.write("SVR resembles linear regression by predicting continuous values but differs by penalizing only errors exceeding a certain margin, rather than all deviations. Kernels distinguish SVR from linear regression: linear kernels model straightforward linear trends, polynomial kernels capture complex interactions, and radial basis function kernels handle intricate nonlinear patterns through similarity measures.")

st.subheader("For each kernel, what happens when you increase or decrease the magnitudes of hyperparameters C and gamma? Why?")
st.write("""
Linear Kernel: 
The C parameter in linear kernels controls the trade-off between fitting the training data and maintaining a large margin. Increasing C reduces regularization, allowing the model to fit training data more closely but risking overfitting. Lower C values enforce more regularization, creating smoother decision boundaries that may underfit the data. Linear kernels are not affected by the gamma parameter.

RBF Kernel:
The C parameter in RBF kernels functions similarly to linear kernels, balancing fit versus regularization. The gamma parameter controls the influence radius of each training example. Lower gamma values create smoother decision boundaries as each example affects a wider area, potentially causing underfitting. Higher gamma values restrict each example's influence to nearby points, creating more complex boundaries that closely follow training data and may overfit. RBF kernels show the most dramatic performance changes with gamma variations.

Polynomial Kernel:
For polynomial kernels, C similarly balances fitting versus regularization. The gamma parameter influences the curvature and complexity of the decision boundary. Higher gamma values create more curved, complex boundaries that may overfit the data. Polynomial kernels generally show less sensitivity to extreme gamma values compared to RBF kernels.
         """)

st.image("streamlitImages/svrplots.png")

st.header("Decision Tree Regressor")

st.subheader("How does a DT regressor work? ")
st.write("Decision Tree regressors work by splitting the dataset based on the values of the features that best reduce prediction error. At each split, a feature and error threshold is selected that shows the greatest error reduction, which divded the data into more homogenous groups. This process continues until it reaches a stopping point such as maximum depth, or the algorithm doesn't sense any further improvement in error.")

st.subheader("What happens when you increase or decrease the maximum depth?")
st.write("When you increase the maximum depth, it allows the tree to fit the training data more closely. However this can lead to a higher test error and overfitting the model. When you decrease the maximum depth, it simplifies the model and allows for less splits and improves generalization on unseen data. However, this may underfit the model.")
st.image("streamlitImages/dt1.png")

st.subheader("What happens when you reduce the number of features?")
st.write("When you reduce the number of features, it can help make a more simple model and have a faster train time. However, this may lead to a loss of predictive power if you remove important features. Depending on the feature (important vs irrelevant), the model could perform better due to removing overfitting for removing irrelevant features. ")
st.image("streamlitImages/dt2.png")

st.subheader("What happens when you normalize features? If features are already normalized, what happens when you scale them out of proportion?")
st.write("For decision trees, normalization doesn't have a major effect. Since decision trees split based on feature error rather than distance, the effect is very small. However, if one feature is scaled out incredibly far, it may take over the split decisions and reduce the model's overall accuracy. ")
st.image("streamlitImages/dt3.png")