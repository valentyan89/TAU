from pathlib import Path
import re

root = Path(r"C:\Users\Valenty`an\AndroidStudioProjects\HaveTime\android")
base = root / 'app/src/main/kotlin/com/example/havetime/presentation'
def read(p): return (base/p).read_text(encoding='utf-8')
def write(p,s):
    target=base/p
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(s,encoding='utf-8')
def edit(p,fn): write(p,fn(read(p)))

write('navigation/Screen.kt','''package com.example.havetime.presentation.navigation

import androidx.navigation3.runtime.NavKey
import kotlinx.serialization.Serializable

@Serializable
sealed class Screen : NavKey {
    @Serializable
    data object Day : Screen()

    @Serializable
    data object Month : Screen()

    @Serializable
    data class Year(val year: Int) : Screen()

    @Serializable
    data object Map : Screen()

    @Serializable
    data object Auth : Screen()

    @Serializable
    data object Settings : Screen()
}
''')
write('navigation/BackStackExtensions.kt','''package com.example.havetime.presentation.navigation

import androidx.navigation3.runtime.NavKey

/** Opens a destination once; an existing destination is revealed instead of duplicated. */
fun MutableList<NavKey>.navigate(screen: Screen) {
    if (lastOrNull() == screen) return
    if (!popTo(screen)) add(screen)
}

/** Returns false at the root so the activity can handle Back normally. */
fun MutableList<NavKey>.goBack(): Boolean {
    if (size <= 1) return false
    removeAt(lastIndex)
    return true
}

/** Removes destinations above the target without removing the target itself. */
fun MutableList<NavKey>.popTo(screen: Screen): Boolean {
    val index = indexOfLast { it == screen }
    if (index < 0) return false
    while (lastIndex > index) removeAt(lastIndex)
    return true
}

fun MutableList<NavKey>.replaceTop(screen: Screen) {
    if (lastOrNull() == screen) return
    if (isNotEmpty()) removeAt(lastIndex)
    navigate(screen)
}
''')

# UI receives callbacks instead of a navigation-library controller.
for p in ['screens/calendar/day_week/DayScreen.kt','screens/calendar/day_week/DayScreenContent.kt','screens/map/MapScreen.kt']:
    def change(s):
        s=s.replace('import androidx.navigation.NavController\n','')
        s=s.replace('    navController: NavController,','    onOpenMonth: () -> Unit,')
        s=s.replace('navController = navController,','onOpenMonth = onOpenMonth,')
        s=s.replace('navController.navigate(Screen.Month.route)','onOpenMonth()')
        s=s.replace('Screen.Day.route','Screen.Day').replace('Screen.Map.route','Screen.Map')
        return s
    edit(p,change)

for p in ['screens/calendar/month/MonthScreen.kt','screens/calendar/year/YearScreen.kt','screens/auth/AuthScreen.kt']:
    def change(s):
        s=s.replace('import androidx.navigation.NavController\n','')
        s=s.replace('import com.example.havetime.presentation.common.navigation.bottom_bar.GlassyBottomBar\n','')
        s=s.replace('    navController: NavController,\n','')
        # Remove screen-local bottom bars: the host owns the only bottom bar.
        s=re.sub(r',\n        bottomBar = \{\n            GlassyBottomBar\(.*?\n        \}', '', s, flags=re.S)
        if '/auth/' not in p:
            s=s.replace('    val hazeState = remember { HazeState() }','    val hazeState = LocalSharedBottomBar.current?.hazeState ?: remember { HazeState() }')
            s=s.replace('import dev.chrisbanes.haze.HazeState','import com.example.havetime.presentation.common.navigation.bottom_bar.LocalSharedBottomBar\nimport dev.chrisbanes.haze.HazeState')
        if '/year/' in p:
            s=s.replace('    sharedViewModel: MonthViewModel,','    sharedViewModel: MonthViewModel,\n    onOpenMonth: () -> Unit,')
            s=s.replace('navController.navigate(Screen.Month.route)','onOpenMonth()')
            s=s.replace('import dev.chrisbanes.haze.HazeState','import androidx.compose.ui.graphics.Color\nimport androidx.compose.ui.unit.dp\nimport dev.chrisbanes.haze.haze\nimport dev.chrisbanes.haze.HazeState')
            s=s.replace('                    Modifier\n                        .fillMaxSize()', '''                    Modifier
                        .haze(
                            hazeState,
                            backgroundColor = MaterialTheme.colorScheme.background,
                            tint = Color.Black.copy(alpha = .2f),
                            blurRadius = 30.dp
                        )
                        .fillMaxSize()''')
        return s
    edit(p,change)

for p in ['screens/settings/SettingsScreen.kt','screens/settings/SettingsScreenContent.kt']:
    edit(p,lambda s:s.replace('import androidx.navigation.NavController\n','').replace('    navController: NavController,\n','').replace('        navController = navController,\n',''))

edit('screens/calendar/month/components/MonthBottomNavigationBar.kt',lambda s:s.replace('import androidx.navigation.NavController\n','').replace('navController: NavController','onOpenMap: () -> Unit').replace('{ navController.navigate(Screen.Map.route) }','onOpenMap'))

p='common/navigation/drawer/CalendarDrawerContent.kt'
s=read(p).replace('import androidx.navigation.NavController\n','')
s=s.replace('    navController: NavController,\n    currentRoute: String,','    currentScreen: Screen,\n    onNavigate: (Screen) -> Unit,')
s=s.replace('currentRoute.startsWith(Screen.Year.route)','currentScreen is Screen.Year')
s=re.sub(r'currentRoute == Screen\.(\w+)\.route',r'currentScreen == Screen.\1',s)
s=re.sub(r'navController.navigate\(Screen\.(\w+)\.route\) \{.*?\n                    \}',r'onNavigate(Screen.\1)',s,flags=re.S)
s=re.sub(r'navController.navigate\("\$\{Screen.Year.route\}/\$currentYear"\) \{.*?\n                    \}',r'onNavigate(Screen.Year(currentYear))',s,flags=re.S)
write(p,s)

p='common/navigation/bottom_bar/GlassyBottomBar.kt'
s=read(p).replace('import androidx.navigation.NavController\n','').replace('import androidx.navigation.compose.currentBackStackEntryAsState\n','')
s=s.replace('    navController: NavController,','    currentScreen: Screen,\n    onNavigate: (Screen) -> Unit,')
s=s.replace('    val navBackStackEntry by navController.currentBackStackEntryAsState()\n    val currentRoute = navBackStackEntry?.destination?.route\n','')
s=s.replace('currentRoute','currentScreen')
s=re.sub(r'Screen\.(\w+)\.route',r'Screen.\1',s)
s=re.sub(r'navController.navigate\(target\) \{.*?\n                                \}', 'onNavigate(target)',s,flags=re.S)
write(p,s)

p='common/navigation/bottom_bar/SharedBottomBarHost.kt'
s=read(p).replace('import androidx.navigation.NavController\n','')
s=s.replace('mutableStateMapOf<String, () -> Unit>()','mutableStateMapOf<Screen, () -> Unit>()')
s=s.replace('route: String','screen: Screen').replace('host, route','host, screen').replace('addActions[route]','addActions[screen]').replace('remove(route)','remove(screen)')
s=s.replace('    navController: NavController,\n    currentRoute: String,','    currentScreen: Screen,\n    onNavigate: (Screen) -> Unit,')
s=s.replace('currentRoute','currentScreen')
s=re.sub(r'Screen\.(\w+)\.route',r'Screen.\1',s)
s=s.replace('    val showBar = showAdd || currentScreen == Screen.Settings','    val showBar = currentScreen != Screen.Auth')
s=s.replace('navController = navController,','currentScreen = currentScreen,\n                    onNavigate = onNavigate,')
write(p,s)

write('navigation/CalendarNavHost.kt','''package com.example.havetime.presentation.navigation

import androidx.activity.compose.BackHandler
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.rememberDrawerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.rememberCoroutineScope
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.viewmodel.navigation3.rememberViewModelStoreNavEntryDecorator
import androidx.navigation3.runtime.entryProvider
import androidx.navigation3.runtime.rememberNavBackStack
import androidx.navigation3.runtime.rememberSaveableStateHolderNavEntryDecorator
import androidx.navigation3.ui.NavDisplay
import com.example.havetime.presentation.common.navigation.bottom_bar.SharedBottomBarHost
import com.example.havetime.presentation.common.navigation.drawer.CalendarDrawerContent
import com.example.havetime.presentation.screens.auth.AuthScreen
import com.example.havetime.presentation.screens.calendar.day_week.DayScreen
import com.example.havetime.presentation.screens.calendar.month.MonthScreen
import com.example.havetime.presentation.screens.calendar.month.MonthViewModel
import com.example.havetime.presentation.screens.calendar.year.YearScreen
import com.example.havetime.presentation.screens.calendar.year.YearViewModel
import com.example.havetime.presentation.screens.map.MapScreen
import com.example.havetime.presentation.screens.settings.SettingsScreen
import kotlinx.coroutines.launch

@Composable
fun CalendarNavHost() {
    val backStack = rememberNavBackStack(Screen.Day)
    val currentScreen = backStack.last() as Screen
    val drawerState = rememberDrawerState(initialValue = DrawerValue.Closed)
    val scope = rememberCoroutineScope()
    // Shared across destinations; individual screen ViewModels belong to NavEntries.
    val sharedViewModel: MonthViewModel = hiltViewModel()
    val openMenu: () -> Unit = { scope.launch { drawerState.open() } }
    val openAuth: () -> Unit = { backStack.navigate(Screen.Auth) }
    val openMonth: () -> Unit = { backStack.navigate(Screen.Month) }

    ModalNavigationDrawer(
        drawerState = drawerState,
        gesturesEnabled = drawerState.isOpen,
        drawerContent = {
            CalendarDrawerContent(
                currentScreen = currentScreen,
                onNavigate = { backStack.navigate(it) },
                onCloseDrawer = { scope.launch { drawerState.close() } }
            )
        }
    ) {
        SharedBottomBarHost(
            currentScreen = currentScreen,
            onNavigate = { backStack.navigate(it) }
        ) {
            NavDisplay(
                backStack = backStack,
                onBack = { backStack.goBack() },
                entryDecorators = listOf(
                    rememberSaveableStateHolderNavEntryDecorator(),
                    rememberViewModelStoreNavEntryDecorator()
                ),
                entryProvider = entryProvider {
                    entry<Screen.Day> {
                        DayScreen(
                            onOpenMonth = openMonth,
                            sharedViewModel = sharedViewModel,
                            onAvatarClick = openAuth,
                            onMenuClick = openMenu
                        )
                    }
                    entry<Screen.Month> {
                        MonthScreen(
                            monthViewModel = sharedViewModel,
                            onAvatarClick = openAuth,
                            onDayClick = { date ->
                                sharedViewModel.selectDate(date)
                                backStack.navigate(Screen.Day)
                            },
                            onYearClick = { backStack.navigate(Screen.Year(it)) },
                            onMenuClick = openMenu
                        )
                    }
                    entry<Screen.Year> { screen ->
                        val viewModel: YearViewModel = hiltViewModel()
                        YearScreen(
                            sharedViewModel = sharedViewModel,
                            onOpenMonth = openMonth,
                            viewModel = viewModel,
                            onAvatarClick = openAuth,
                            onMenuClick = openMenu
                        )
                    }
                    entry<Screen.Map> {
                        MapScreen(
                            onOpenMonth = openMonth,
                            sharedViewModel = sharedViewModel,
                            onAvatarClick = openAuth,
                            onMenuClick = openMenu
                        )
                    }
                    entry<Screen.Auth> {
                        AuthScreen(onAuthSuccess = {}, onBackClick = { backStack.goBack() })
                    }
                    entry<Screen.Settings> {
                        SettingsScreen(onBackClick = { backStack.goBack() })
                    }
                }
            )
        }
    }
    // Registered after NavDisplay so an open drawer handles Back first.
    BackHandler(enabled = drawerState.isOpen) {
        scope.launch { drawerState.close() }
    }
}
''')

# Read the typed year once when the entry ViewModel is first created, avoiding
# an asynchronous current-year load overwriting the requested year.
p='screens/calendar/year/YearViewModel.kt'
s=read(p).replace('import androidx.lifecycle.ViewModel','import androidx.lifecycle.SavedStateHandle\nimport androidx.lifecycle.ViewModel',1)
s=s.replace('    private val getCurrentYearUseCase: GetCurrentYearUseCase,','    private val savedStateHandle: SavedStateHandle,\n    private val getCurrentYearUseCase: GetCurrentYearUseCase,')
s=s.replace('    private val _currentYear = MutableStateFlow(getInitialDateUseCase().year)','    private val _currentYear = MutableStateFlow(savedStateHandle.get<Int>("year") ?: getInitialDateUseCase().year)')
s=s.replace('        loadCurrentYear()','        loadYearData(_currentYear.value)',1)
s=s.replace('    fun setYear(year: Int) {','    fun setYear(year: Int) {\n        savedStateHandle["year"] = year')
write(p,s)
# Supply route arguments through CreationExtras via a small factory-independent
# initialization method, called before screen UI, exactly once per ViewModel.
edit(p,lambda s:s.replace('    fun setYear(year: Int) {','    private var routeInitialized = false\n\n    fun initializeYear(year: Int) {\n        if (routeInitialized) return\n        routeInitialized = true\n        if (savedStateHandle.get<Int>("year") == null) setYear(year)\n    }\n\n    fun setYear(year: Int) {'))
edit('navigation/CalendarNavHost.kt',lambda s:s.replace('                        YearScreen(','                        androidx.compose.runtime.LaunchedEffect(viewModel) {\n                            viewModel.initializeYear(screen.year)\n                        }\n                        YearScreen('))

# Old Navigation Compose artifact is no longer directly used. Hilt's helper
# remains compatible with the ViewModelStoreOwner supplied by the decorator.
p=root/'app/build.gradle.kts'
s=p.read_text(encoding='utf-8').replace('    implementation(libs.androidx.navigation.compose)\n','')
p.write_text(s,encoding='utf-8')
print('Navigation migration written. Remaining legacy navigation references:')
for p in base.rglob('*.kt'):
    s=p.read_text(encoding='utf-8')
    if 'NavController' in s or '.route' in s or 'navController' in s:
        print(p.relative_to(base))
