import numpy as np
import pandas as pd
from PyCROSL.AbsObjectiveFunc import AbsObjectiveFunc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, mean_absolute_error

class ml_prediction(AbsObjectiveFunc):
    """
    Class for performing machine learning prediction as an objective function
    for optimization using the CRO-SL algorithm.
    """
    def __init__(self, size, pred_dataframe, target_dataset, train_indices, test_indices, indiv_file):
        self.size = size
        self.opt = "min"  # it can be "max" or "min"
        
        # Store necessary data references for prediction
        self.pred_dataframe = pred_dataframe
        self.target_dataset = target_dataset
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.indiv_file = indiv_file
        
        # Set up limits for the optimization variables
        self.sup_lim = np.append(np.append(
            np.repeat(60, pred_dataframe.shape[1]),  # Time sequences
            np.repeat(180, pred_dataframe.shape[1])),  # Time lags
            np.repeat(1, pred_dataframe.shape[1]))  # Variable selection
            
        self.inf_lim = np.append(np.append(
            np.repeat(1, pred_dataframe.shape[1]),  # Min time sequences
            np.repeat(0, pred_dataframe.shape[1])),  # Min time lags
            np.repeat(0, pred_dataframe.shape[1]))  # Min variable selection (0/1)
            
        # Initialize the parent class
        super().__init__(self.size, self.opt, self.sup_lim, self.inf_lim)

    def objective(self, solution):
        # Read solution tracking file
        try:
            sol_file = pd.read_csv(self.indiv_file, sep=' ', header=0)
        except:
            sol_file = pd.DataFrame(columns=['CV', 'Test', 'Sol'])
        
        # Parse the solution vector into its components
        time_sequences = np.array(solution[:self.pred_dataframe.shape[1]]).astype(int)
        time_lags = np.array(solution[self.pred_dataframe.shape[1]:2*self.pred_dataframe.shape[1]]).astype(int)
        variable_selection = np.array(solution[2*self.pred_dataframe.shape[1]:]).astype(int)
        
        # Don't proceed if no variables are selected
        if sum(variable_selection) == 0:
            return 100000

        # Create dataset according to solution
        dataset_opt = self.target_dataset.copy()
        for i, col in enumerate(self.pred_dataframe.columns):
            if variable_selection[i] == 0 or time_sequences[i] == 0:
                continue
            for j in range(time_sequences[i]):
                dataset_opt[f"{col}_lag{time_lags[i]+j}"] = self.pred_dataframe[col].shift(time_lags[i]+j)
        
        # Split dataset into train and test
        train_dataset = dataset_opt[self.train_indices]
        test_dataset = dataset_opt[self.test_indices]
        
        # Extract features and target
        Y_column = 'Target'
        X_train = train_dataset[train_dataset.columns.drop([Y_column])]
        Y_train = train_dataset[Y_column]
        X_test = test_dataset[test_dataset.columns.drop([Y_column])]
        Y_test = test_dataset[Y_column]
        
        # Standardize data
        from sklearn import preprocessing
        scaler = preprocessing.StandardScaler()
        X_std_train = scaler.fit_transform(X_train)
        X_std_test = scaler.transform(X_test)
        
        X_train = pd.DataFrame(X_std_train, columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(X_std_test, columns=X_test.columns, index=X_test.index)

        # Train and evaluate model
        clf = LogisticRegression()
        
        # Cross-validation on training set
        from sklearn.model_selection import cross_val_score
        score = cross_val_score(clf, X_train, Y_train, cv=5, scoring='f1')
        
        # Train on full training set and evaluate on test set
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        
        # Print and store results
        print(f"CV F1: {score.mean()}, Test F1: {f1_score(Y_pred, Y_test)}")
        
        # Update solution file
        sol_file = pd.concat([sol_file, pd.DataFrame({
            'CV': [score.mean()], 
            'Test': [f1_score(Y_pred, Y_test)], 
            'Sol': [str(solution)]})], 
            ignore_index=True)
        sol_file.to_csv(self.indiv_file, sep=' ', header=True, index=None)
        
        return 1/score.mean()

    def random_solution(self):
        """Generate a random solution within the defined limits."""
        return np.random.choice(self.sup_lim[0], self.size, replace=True)

    def repair_solution(self, solution):
        """Ensure solution values stay within bounds."""
        return np.clip(solution, self.inf_lim, self.sup_lim)