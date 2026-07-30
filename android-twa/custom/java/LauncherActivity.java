/*
 * DALMUTI's Browser Helper launcher customization.
 *
 * Bubblewrap normally forwards a static Web Manifest orientation value to
 * Chrome. This launcher instead reads Android's auto-rotate setting for every
 * launch. When rotation is locked it pins the TWA to the currently visible
 * portrait/landscape family; when enabled it allows sensor rotation.
 */
package lab.dclab.dalmuti;

import android.content.pm.ActivityInfo;
import android.content.res.Configuration;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;

import androidx.browser.customtabs.CustomTabsCallback;
import androidx.browser.trusted.ScreenOrientation;
import androidx.browser.trusted.TrustedWebActivityIntentBuilder;

import com.google.androidbrowserhelper.trusted.SessionStore;
import com.google.androidbrowserhelper.trusted.SharedPreferencesTokenStore;
import com.google.androidbrowserhelper.trusted.TwaLauncher;
import com.google.androidbrowserhelper.trusted.TwaLauncher.FallbackStrategy;
import com.google.androidbrowserhelper.trusted.splashscreens.SplashScreenStrategy;

public class LauncherActivity
        extends com.google.androidbrowserhelper.trusted.LauncherActivity {

    private boolean isSystemAutoRotateEnabled() {
        try {
            return Settings.System.getInt(
                    getContentResolver(),
                    Settings.System.ACCELEROMETER_ROTATION,
                    0
            ) == 1;
        } catch (SecurityException ignored) {
            // Conservatively preserve the current orientation if an OEM
            // restricts access to this otherwise public system preference.
            return false;
        }
    }

    private int currentTwaOrientation() {
        int orientation = getResources().getConfiguration().orientation;
        if (orientation == Configuration.ORIENTATION_LANDSCAPE) {
            return ScreenOrientation.LANDSCAPE;
        }
        if (orientation == Configuration.ORIENTATION_PORTRAIT) {
            return ScreenOrientation.PORTRAIT;
        }
        return getResources().getDisplayMetrics().widthPixels
                        > getResources().getDisplayMetrics().heightPixels
                ? ScreenOrientation.LANDSCAPE
                : ScreenOrientation.PORTRAIT;
    }

    private int requestedTwaOrientation() {
        return isSystemAutoRotateEnabled()
                ? ScreenOrientation.ANY
                : currentTwaOrientation();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        // A requested orientation and a translucent launch theme can crash on
        // Android 8.0. Browser Helper has the same API 26 guard.
        if (Build.VERSION.SDK_INT > Build.VERSION_CODES.O) {
            setRequestedOrientation(
                    isSystemAutoRotateEnabled()
                            ? ActivityInfo.SCREEN_ORIENTATION_FULL_USER
                            : ActivityInfo.SCREEN_ORIENTATION_LOCKED
            );
        }
        super.onCreate(savedInstanceState);
    }

    @Override
    protected TwaLauncher createTwaLauncher() {
        final int launchOrientation = requestedTwaOrientation();
        return new TwaLauncher(
                this,
                null,
                SessionStore.makeSessionId(getTaskId()),
                new SharedPreferencesTokenStore(this)
        ) {
            @Override
            public void launch(
                    TrustedWebActivityIntentBuilder builder,
                    CustomTabsCallback customTabsCallback,
                    SplashScreenStrategy splashScreenStrategy,
                    Runnable completionCallback,
                    FallbackStrategy fallbackStrategy
            ) {
                builder.setScreenOrientation(launchOrientation);
                super.launch(
                        builder,
                        customTabsCallback,
                        splashScreenStrategy,
                        completionCallback,
                        fallbackStrategy
                );
            }
        };
    }
}
