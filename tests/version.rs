use mlmodelc_export::VERSION;

#[test]
fn version_constant_is_the_package_version() {
    let manifest = include_str!("../Cargo.toml");
    let package = manifest.split("[package]").nth(1).unwrap();
    let declared = package
        .lines()
        .find_map(|line| line.strip_prefix("version = "))
        .unwrap()
        .trim_matches('"');
    assert_eq!(VERSION, declared);
    let parts: Vec<_> = VERSION.split('.').collect();
    assert_eq!(parts.len(), 3, "{VERSION}");
    assert!(
        parts.iter().all(|part| part.parse::<u32>().is_ok()),
        "{VERSION}"
    );
}
