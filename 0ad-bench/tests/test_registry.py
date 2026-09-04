from zero_ad_bench import TaskRegistry


def test_v01_catalog_has_fifty_balanced_tasks():
    registry = TaskRegistry.default()
    summary = registry.summary()
    assert summary["task_count"] == 50
    assert summary["categories"] == {
        "adaptive": 10,
        "building-tech": 10,
        "combat": 10,
        "economy": 10,
        "scouting": 10,
    }
    assert summary["tracks"] == {"hybrid": 50, "symbolic": 50, "vision": 18}
    assert summary["splits"] == {"development": 45, "evaluation": 5}


def test_registry_filters_and_orders_tasks():
    tasks = TaskRegistry.default().list(category="economy", track="symbolic")
    assert len(tasks) == 10
    assert [task.id for task in tasks] == sorted(task.id for task in tasks)
    assert tasks[0].id == "econ-001"
