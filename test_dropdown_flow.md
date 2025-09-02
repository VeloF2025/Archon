# Test Dropdown Flow for API Keys

## How to Test the Dropdown Feature

### 1. Access the Settings Page
- Navigate to http://localhost:3737
- Click on the Settings tab (gear icon)
- Go to the "API Keys" section

### 2. Add a New API Key with Dropdown
- Click the "Add Credential" button
- You should now see:
  - A **dropdown** in the first column showing provider options (OpenAI, DeepSeek, etc.)
  - Below the dropdown, it shows "Key: OPENAI_API_KEY" (or the selected provider's key)
  - A text field for entering the API key value
  - A shield icon to mark it as validator
  - A delete icon

### 3. Test Provider Selection
- Select "DeepSeek" from the dropdown
- The key name should change to "Key: DEEPSEEK_API_KEY"
- Select "OpenAI" from the dropdown  
- The key name should change to "Key: OPENAI_API_KEY"

### 4. Save with Provider Metadata
- Enter your API key in the value field
- Click the shield icon to mark it for validator use (it should turn purple)
- Click "Save All Changes"
- The credential should be saved with:
  - The correct key name based on provider
  - The provider stored in metadata
  - The useAsValidator flag if selected

### 5. Verify Validator Picks Up Provider
- Check validator logs:
```bash
docker compose logs archon-validator --tail=50
```
- Look for: "Using provider from metadata: deepseek" (or your selected provider)
- This confirms the validator is using the dropdown selection

## Expected Behavior

✅ **New Credentials**: Show dropdown for provider selection
✅ **Existing Credentials**: Show disabled text input with key name
✅ **Provider Metadata**: Saved to Supabase with credential
✅ **Validator Detection**: Uses provider from metadata (not key format guessing)
✅ **Key Names**: Automatically set based on provider selection

## Troubleshooting

If dropdown doesn't appear:
1. Hard refresh the page (Ctrl+Shift+R)
2. Clear browser cache
3. Check browser console for errors
4. Restart frontend container: `docker compose restart archon-frontend`

If validator doesn't pick up provider:
1. Check metadata is saved: Look in Supabase credentials table
2. Restart validator: `docker compose --profile validator restart archon-validator`
3. Check logs for "Using provider from metadata"