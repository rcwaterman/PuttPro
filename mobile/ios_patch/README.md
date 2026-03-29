# iOS Build Requirements

## Prerequisites (all macOS-only)
- macOS 13+ (Ventura or later recommended)
- Xcode 15+ — install from the Mac App Store
- Xcode Command Line Tools: `xcode-select --install`
- CocoaPods: `sudo gem install cocoapods`
- Node.js: `brew install node`

## One-time setup (run from mobile/)
```bash
npm install
npm run prepare
npx cap add ios
bash ios_patch/apply.sh      # patches Info.plist
npx cap sync
npx cap open ios             # opens Xcode
```

## In Xcode
1. Select your Apple Developer signing team in *Signing & Capabilities*
2. Connect your iPhone via USB (or use simulator for basic testing)
3. Build & Run (⌘R)

## Certificate trust on iOS
After installing the app:
1. Start the PC server — note the CA cert QR code in startup output
2. Scan the CA cert QR on your iPhone to download `PuttPro-CA.crt`
3. Settings → General → VPN & Device Management → install the profile
4. Settings → General → About → Certificate Trust Settings → enable full trust
5. Reopen PuttPro — connection will be trusted

## Iterating
After changing `mobile/src/index.html`:
```bash
npx cap sync && npx cap open ios
```
Then rebuild in Xcode.

## Production notes
- Remove `NSAllowsLocalNetworking` from Info.plist before App Store submission
- Use a real domain + valid TLS cert for the cloud server
- Add proper user auth (login flow) before shipping to end users
