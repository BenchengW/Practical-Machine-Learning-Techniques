import pandas as pd
from pandas.testing import assert_frame_equal
from src.transformers import GoalAdjustor, TimeTransformer


def test_time_transformer():
    time_transformer = TimeTransformer()
    deadline_timestamp = 1459283229
    created_at_timestamp = 1455845363
    launched_at_timestamp = 1456694829
    sample_df = pd.DataFrame({'deadline': [deadline_timestamp], 'created_at': [created_at_timestamp], 'launched_at': [
        launched_at_timestamp]})

    expected_df = pd.DataFrame({'launched_to_deadline': [29], 'created_to_launched': [9]})

    result_df = time_transformer.transform(sample_df)

    assert_frame_equal(result_df, expected_df)
    # assert result_df.equals(expected_df)

