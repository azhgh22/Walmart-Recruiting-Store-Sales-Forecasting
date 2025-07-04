from sklearn.base import BaseEstimator, RegressorMixin, clone

class GroupStatModel(BaseEstimator, RegressorMixin):
    """
    A flexible model that predicts based on hierarchical groupings (Store, Dept, Global),
    using externally provided pipelines for each stage.
    """

    def __init__(self, store_pipeline, dept_pipeline, global_pipeline):
        self.store_pipeline = clone(store_pipeline)
        self.dept_pipeline = clone(dept_pipeline)
        self.global_pipeline = clone(global_pipeline)

    def _process_group(self, X, y, groupby_cols):
      from feature_engineering.grouper import group_and_aggregate
      return group_and_aggregate(X, y, groupby_cols=groupby_cols, y_aggfunc='mean')

    def fit(self, X, y):
        # Store model stage
        X_store, y_store = self._process_group(
            X, y, groupby_cols=['Store', 'Date', 'IsHoliday']
        )
        
        self.store_pipeline.fit(X_store, y_store)
        store_pred = self.store_pipeline.predict(X_store)

        store_results = X_store[['Store', 'Date']].copy()
        store_results['Store_Prediction'] = store_pred

        # Dept model stage
        X_dept, y_dept = self._process_group(
            X, y, groupby_cols=['Date', 'IsHoliday', 'Dept']
        )
        self.dept_pipeline.fit(X_dept, y_dept)
        dept_pred = self.dept_pipeline.predict(X_dept)

        dept_results = X_dept[['Dept', 'Date']].copy()
        dept_results['Dept_Prediction'] = dept_pred

        # Global model stage
        X_global = X.copy()
        X_global = X_global.merge(dept_results, on=['Date', 'Dept'], how='left')
        X_global = X_global.merge(store_results, on=['Date', 'Store'], how='left')

        

        self.global_pipeline.fit(X_global, y)

        return self

    def predict(self, X):
        y_dummy = None

        X_store, _ = self._process_group(
            X, y_dummy, groupby_cols=['Store', 'Date', 'IsHoliday']
        )
        store_pred = self.store_pipeline.predict(X_store)
        store_results = X_store[['Store', 'Date']].copy()
        store_results['Store_Prediction'] = store_pred

        # Dept stage
        X_dept, _ = self._process_group(
            X, y_dummy, groupby_cols=['Date', 'IsHoliday', 'Dept']
        )
        dept_pred = self.dept_pipeline.predict(X_dept)
        dept_results = X_dept[['Dept', 'Date']].copy()
        dept_results['Dept_Prediction'] = dept_pred

        # Global stage
        X_global = X.copy()
        X_global = X_global.merge(dept_results, on=['Date', 'Dept'], how='left')
        X_global = X_global.merge(store_results, on=['Date', 'Store'], how='left')

        return self.global_pipeline.predict(X_global)
