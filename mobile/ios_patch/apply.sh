#!/bin/bash
# Applies iOS-specific patches after 'npx cap add ios'.
# Run from the mobile/ directory on macOS:
#   bash ios_patch/apply.sh

set -e

PLIST="ios/App/App/Info.plist"

if [ ! -f "$PLIST" ]; then
  echo "Error: $PLIST not found."
  echo "Run 'npx cap add ios' first, then re-run this script."
  exit 1
fi

echo "Patching $PLIST ..."

# Camera permission — required by App Store and getUserMedia
/usr/libexec/PlistBuddy -c \
  "Add :NSCameraUsageDescription string 'PuttPro needs camera access to stream disc flight video to the analysis server.'" \
  "$PLIST" 2>/dev/null || \
/usr/libexec/PlistBuddy -c \
  "Set :NSCameraUsageDescription 'PuttPro needs camera access to stream disc flight video to the analysis server.'" \
  "$PLIST"

# App Transport Security — allow plain HTTP to the CA cert download server (port 5001)
# and arbitrary local network addresses for development.
# For production cloud deployment remove NSAllowsLocalNetworking and add your domain.
/usr/libexec/PlistBuddy -c \
  "Add :NSAppTransportSecurity dict" "$PLIST" 2>/dev/null || true

/usr/libexec/PlistBuddy -c \
  "Add :NSAppTransportSecurity:NSAllowsLocalNetworking bool true" "$PLIST" 2>/dev/null || \
/usr/libexec/PlistBuddy -c \
  "Set :NSAppTransportSecurity:NSAllowsLocalNetworking true" "$PLIST"

echo "Done. Info.plist patched."
echo ""
echo "Next steps:"
echo "  npx cap sync"
echo "  npx cap open ios"
echo "  In Xcode: select your signing team, then build & run."
