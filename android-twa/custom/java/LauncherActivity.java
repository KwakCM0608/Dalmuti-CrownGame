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
import android.graphics.Color;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.Settings;
import android.view.View;
import android.view.animation.DecelerateInterpolator;
import android.widget.FrameLayout;
import android.widget.ImageView;

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

    // 75 + 825 ms reveal, a short settled beat, then a 180 ms Chrome fade.
    // The native presentation budget remains comfortably below 1.5 seconds.
    private static final long SPLASH_REVEAL_DELAY_MS = 75L;
    private static final long SPLASH_REVEAL_DURATION_MS = 825L;
    private static final long SPLASH_HANDOFF_AT_MS = 950L;
    private static final String NATIVE_LAUNCH_DISPATCHED_KEY =
            "dalmuti.nativeLaunchDispatched";

    private final Handler splashHandler = new Handler(Looper.getMainLooper());
    private boolean nativeLaunchDispatched;

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

    private ImageView splashLayer(int drawableId) {
        ImageView layer = new ImageView(this);
        layer.setImageResource(drawableId);
        layer.setScaleType(ImageView.ScaleType.FIT_CENTER);
        layer.setLayoutParams(
                new FrameLayout.LayoutParams(
                        FrameLayout.LayoutParams.MATCH_PARENT,
                        FrameLayout.LayoutParams.MATCH_PARENT
                )
        );
        return layer;
    }

    private void showNativeSplash() {
        getWindow().setStatusBarColor(Color.BLACK);
        getWindow().setNavigationBarColor(Color.BLACK);

        FrameLayout root = new FrameLayout(this);
        root.setBackgroundColor(Color.BLACK);

        ImageView glow = splashLayer(R.drawable.dalmuti_splash_glow_v4);
        glow.setAlpha(0f);
        glow.setScaleX(0.90f);
        glow.setScaleY(0.90f);

        ImageView artwork = splashLayer(R.drawable.dalmuti_splash_v4);
        artwork.setAlpha(0f);
        artwork.setScaleX(0.94f);
        artwork.setScaleY(0.94f);

        root.addView(glow);
        root.addView(artwork);
        setContentView(root);

        glow.setLayerType(View.LAYER_TYPE_HARDWARE, null);
        glow.animate()
                .alpha(0.48f)
                .scaleX(1.035f)
                .scaleY(1.035f)
                .setStartDelay(35L)
                .setDuration(865L)
                .setInterpolator(new DecelerateInterpolator(1.5f))
                .withEndAction(
                        () -> glow.setLayerType(View.LAYER_TYPE_NONE, null)
                )
                .start();

        artwork.setLayerType(View.LAYER_TYPE_HARDWARE, null);
        artwork.animate()
                .alpha(1f)
                .scaleX(1f)
                .scaleY(1f)
                .setStartDelay(SPLASH_REVEAL_DELAY_MS)
                .setDuration(SPLASH_REVEAL_DURATION_MS)
                .setInterpolator(new DecelerateInterpolator(1.35f))
                .withEndAction(
                        () -> artwork.setLayerType(View.LAYER_TYPE_NONE, null)
                )
                .start();
    }

    private void launchTwaOnce() {
        if (nativeLaunchDispatched || isFinishing()) {
            return;
        }
        nativeLaunchDispatched = true;
        launchTwa();
    }

    @Override
    protected ImageView.ScaleType getSplashImageScaleType() {
        return ImageView.ScaleType.FIT_CENTER;
    }

    @Override
    protected boolean shouldLaunchImmediately() {
        return false;
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

        nativeLaunchDispatched =
                savedInstanceState != null
                        && savedInstanceState.getBoolean(
                                NATIVE_LAUNCH_DISPATCHED_KEY,
                                false
                        );
        if (isFinishing() || nativeLaunchDispatched) {
            return;
        }

        showNativeSplash();
        splashHandler.postDelayed(this::launchTwaOnce, SPLASH_HANDOFF_AT_MS);
    }

    @Override
    protected void onSaveInstanceState(Bundle outState) {
        super.onSaveInstanceState(outState);
        outState.putBoolean(
                NATIVE_LAUNCH_DISPATCHED_KEY,
                nativeLaunchDispatched
        );
    }

    @Override
    protected void onDestroy() {
        splashHandler.removeCallbacksAndMessages(null);
        super.onDestroy();
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
