from pathlib import Path
import re
root=Path(r"C:\Users\Valenty`an\AndroidStudioProjects\HaveTime\android")
base=root/'app/src/main/kotlin/com/example/havetime/presentation'
p=base/'screens/calendar/year/YearViewModel.kt'
s=p.read_text(encoding='utf-8')
s=s.replace('import com.example.havetime.domain.usecase.date.GetCurrentYearUseCase\n','')
s=s.replace('    private val getCurrentYearUseCase: GetCurrentYearUseCase,\n','')
s=re.sub(r'    init \{\n        loadYearData\(_currentYear.value\)\n    \}\n\n    private fun loadCurrentYear\(\) \{.*?\n    \}\n\n', '',s,flags=re.S)
s=s.replace('        if (savedStateHandle.get<Int>("year") == null) setYear(year)', '''        val initialYear = savedStateHandle.get<Int>("year") ?: year
        savedStateHandle["year"] = initialYear
        _currentYear.value = initialYear
        loadYearData(initialYear)''')
s=s.replace('        _currentYear.value = nextYear\n        loadYearData(nextYear)','        setYear(nextYear)').replace('        _currentYear.value = prevYear\n        loadYearData(prevYear)','        setYear(prevYear)')
s=s.replace('import kotlinx.coroutines.Dispatchers','import kotlinx.coroutines.CancellationException\nimport kotlinx.coroutines.Job\nimport kotlinx.coroutines.Dispatchers')
s=s.replace('    private fun loadYearData(year: Int) {\n        viewModelScope.launch {','    private var loadJob: Job? = null\n\n    private fun loadYearData(year: Int) {\n        loadJob?.cancel()\n        loadJob = viewModelScope.launch {')
s=s.replace('            } catch (e: Exception) {','            } catch (e: CancellationException) {\n                throw e\n            } catch (e: Exception) {')
p.write_text(s,encoding='utf-8')
p=base/'screens/auth/AuthScreen.kt'
s=p.read_text(encoding='utf-8').replace('import dev.chrisbanes.haze.HazeState\n','').replace('    val hazeState = remember { HazeState() }\n\n','')
p.write_text(s,encoding='utf-8')
test=root/'app/src/test/kotlin/com/example/havetime/presentation/navigation/BackStackExtensionsTest.kt'
test.parent.mkdir(parents=True,exist_ok=True)
test.write_text('''package com.example.havetime.presentation.navigation

import androidx.navigation3.runtime.NavKey
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class BackStackExtensionsTest {
    @Test
    fun repeatedDestinationDoesNotDuplicateEntries() {
        val stack = mutableListOf<NavKey>(Screen.Day)
        stack.navigate(Screen.Map)
        stack.navigate(Screen.Map)
        assertEquals(listOf(Screen.Day, Screen.Map), stack)
    }

    @Test
    fun returningToMonthRemovesYearAndKeepsExistingMonth() {
        val stack = mutableListOf<NavKey>(Screen.Day, Screen.Month, Screen.Year(2026))
        stack.navigate(Screen.Month)
        assertEquals(listOf(Screen.Day, Screen.Month), stack)
    }

    @Test
    fun rootCannotBeRemovedByBack() {
        val stack = mutableListOf<NavKey>(Screen.Day, Screen.Map)
        assertTrue(stack.goBack())
        assertFalse(stack.goBack())
        assertEquals(listOf(Screen.Day), stack)
    }

    @Test
    fun absentPopTargetDoesNotChangeStack() {
        val stack = mutableListOf<NavKey>(Screen.Day, Screen.Map)
        assertFalse(stack.popTo(Screen.Month))
        assertEquals(listOf(Screen.Day, Screen.Map), stack)
    }

    @Test
    fun yearsCarryDistinctArguments() {
        val stack = mutableListOf<NavKey>(Screen.Day)
        stack.navigate(Screen.Year(2026))
        stack.navigate(Screen.Year(2027))
        assertEquals(Screen.Year(2027), stack.last())
        stack.goBack()
        assertEquals(Screen.Year(2026), stack.last())
    }

    @Test
    fun replacementDoesNotDuplicateExistingDestination() {
        val stack = mutableListOf<NavKey>(Screen.Day, Screen.Month, Screen.Auth)
        stack.replaceTop(Screen.Month)
        assertEquals(listOf(Screen.Day, Screen.Month), stack)
    }
}
''',encoding='utf-8')
print('Year initialization and navigation tests updated')
