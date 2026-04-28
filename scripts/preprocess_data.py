import pandas as pd
from datasets import load_dataset
import os

def preprocess(sample_ratio=0.5, seed=3407):
    print("Loading BANKING77 dataset...")
    
    # Links trực tiếp từ HuggingFace
    train_url = "https://huggingface.co/datasets/PolyAI/banking77/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    test_url = "https://huggingface.co/datasets/PolyAI/banking77/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet"

    try:
        df_train = pd.read_parquet(train_url)
        df_test = pd.read_parquet(test_url)
        print("Tải dữ liệu thành công bằng Pandas!")
    except Exception as e:
        print(f"Lỗi tải trực tiếp: {e}")
        print("Đang thử phương án dự phòng (PolyAI/banking77)...")
        ds = load_dataset("PolyAI/banking77")
        df_train = ds['train'].to_pandas()
        df_test = ds['test'].to_pandas()

    # Danh sách 77 intent tiêu chuẩn (fix: transfer_timing thay vì trùng lặp)
    label_names = [
        'activate_my_card', 'age_limit', 'apple_pay_or_google_pay', 'atm_support',
        'automatic_top_up', 'balance_not_updated_after_bank_transfer',
        'balance_not_updated_after_cheque_or_cash_deposit', 'beneficiary_not_allowed',
        'cancel_transfer', 'card_about_to_expire', 'card_acceptance', 'card_arrival',
        'card_delivery_estimate', 'card_linking', 'card_not_working',
        'card_payment_fee_charged', 'card_payment_not_recognised',
        'card_payment_wrong_exchange_rate', 'card_swallowed', 'cash_withdrawal_charge',
        'cash_withdrawal_not_recognised', 'change_pin', 'compromised_card',
        'contactless_not_working', 'country_support', 'declined_card_payment',
        'declined_cash_withdrawal', 'declined_transfer',
        'direct_debit_payment_not_recognised', 'disposable_card_limits',
        'edit_personal_details', 'exchange_charge', 'exchange_rate', 'exchange_via_app',
        'extra_charge_on_statement', 'failed_transfer', 'fiat_currency_support',
        'get_disposable_virtual_card', 'get_physical_card', 'getting_spare_card',
        'getting_virtual_card', 'lost_or_stolen_card', 'lost_or_stolen_phone',
        'order_physical_card', 'passcode_forgotten', 'pending_card_payment',
        'pending_cash_withdrawal', 'pending_top_up', 'pending_transfer', 'pin_blocked',
        'receiving_money', 'refund_not_showing_up', 'request_refund',
        'reverted_card_payment?', 'supported_cards_and_currencies', 'terminate_account',
        'top_up_by_bank_transfer_charge', 'top_up_by_card_charge',
        'top_up_by_cash_or_cheque', 'top_up_failed', 'top_up_limits', 'top_up_reverted',
        'topping_up_by_card', 'transaction_charged_twice', 'transfer_fee_charged',
        'transfer_into_account', 'transfer_not_received_by_recipient',
        'transfer_timing', 'unable_to_verify_identity', 'verify_my_identity',
        'verify_source_of_funds', 'verify_top_up', 'virtual_card_not_working',
        'visa_or_mastercard', 'why_verify_identity', 'wrong_amount_of_cash_received',
        'wrong_exchange_rate_for_cash_withdrawal'
    ]

    assert len(label_names) == 77, f"Expected 77 labels, got {len(label_names)}"

    # Map label ID to label name
    if 'label' in df_train.columns and df_train['label'].dtype != 'O':
        df_train['intent'] = df_train['label'].apply(lambda x: label_names[x] if x < len(label_names) else "unknown")
        df_test['intent'] = df_test['label'].apply(lambda x: label_names[x] if x < len(label_names) else "unknown")
    else:
        df_train['intent'] = df_train.get('intent', df_train.get('label'))
        df_test['intent'] = df_test.get('intent', df_test.get('label'))
    
    # Sample dữ liệu train (stratified by intent)
    print(f"\nFull train set: {len(df_train)} samples")
    df_train_sampled = df_train.groupby('intent', group_keys=False).apply(
        lambda x: x.sample(frac=sample_ratio, random_state=seed)
    ).reset_index(drop=True)
    print(f"Sampled train set ({int(sample_ratio*100)}%): {len(df_train_sampled)} samples")

    # Create sample_data directory
    os.makedirs("sample_data", exist_ok=True)
    
    # Save to CSV
    print("Saving processed data to sample_data/...")
    df_train_sampled[['text', 'intent']].to_csv("sample_data/train.csv", index=False)
    df_test[['text', 'intent']].to_csv("sample_data/test.csv", index=False)
    
    print(f"Dataset prepared: {len(df_train_sampled)} train samples, {len(df_test)} test samples.")

if __name__ == "__main__":
    preprocess(sample_ratio=0.5, seed=3407)
