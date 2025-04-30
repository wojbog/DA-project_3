
path = "data/lectures evaluation.csv"
data = pd.read_csv(path, header=None)
target_map = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1}
num_criteria = 4
data_input = data.iloc[:, :num_criteria]
data_target = data[num_criteria].apply(lambda x: target_map[x])

data_input = data_input.values.reshape(-1, 1, num_criteria)

X_train, X_test, y_train, y_test = train_test_split(
    data_input, data_target.values, test_size=0.2, random_state=1234
)

train_dataloader = CreateDataLoader(X_train, y_train)
test_dataloader = CreateDataLoader(X_test, y_test)
